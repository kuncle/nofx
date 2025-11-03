package decision

import (
	"encoding/json"
	"fmt"
	"log"
	"nofx/market"
	"nofx/mcp"
	"nofx/pool"
	"strings"
	"time"
)

// PositionInfo 持仓信息
type PositionInfo struct {
	Symbol           string  `json:"symbol"`
	Side             string  `json:"side"` // "long" or "short"
	EntryPrice       float64 `json:"entry_price"`
	MarkPrice        float64 `json:"mark_price"`
	Quantity         float64 `json:"quantity"`
	Leverage         int     `json:"leverage"`
	UnrealizedPnL    float64 `json:"unrealized_pnl"`
	UnrealizedPnLPct float64 `json:"unrealized_pnl_pct"`
	LiquidationPrice float64 `json:"liquidation_price"`
	MarginUsed       float64 `json:"margin_used"`
	UpdateTime       int64   `json:"update_time"` // 持仓更新时间戳（毫秒）
}

// AccountInfo 账户信息
type AccountInfo struct {
	TotalEquity      float64 `json:"total_equity"`      // 账户净值
	AvailableBalance float64 `json:"available_balance"` // 可用余额
	TotalPnL         float64 `json:"total_pnl"`         // 总盈亏
	TotalPnLPct      float64 `json:"total_pnl_pct"`     // 总盈亏百分比
	MarginUsed       float64 `json:"margin_used"`       // 已用保证金
	MarginUsedPct    float64 `json:"margin_used_pct"`   // 保证金使用率
	PositionCount    int     `json:"position_count"`    // 持仓数量
}

// CandidateCoin 候选币种（来自币种池）
type CandidateCoin struct {
	Symbol  string   `json:"symbol"`
	Sources []string `json:"sources"` // 来源: "ai500" 和/或 "oi_top"
}

// OITopData 持仓量增长Top数据（用于AI决策参考）
type OITopData struct {
	Rank              int     // OI Top排名
	OIDeltaPercent    float64 // 持仓量变化百分比（1小时）
	OIDeltaValue      float64 // 持仓量变化价值
	PriceDeltaPercent float64 // 价格变化百分比
	NetLong           float64 // 净多仓
	NetShort          float64 // 净空仓
}

// Context 交易上下文（传递给AI的完整信息）
type Context struct {
	CurrentTime     string                  `json:"current_time"`
	RuntimeMinutes  int                     `json:"runtime_minutes"`
	CallCount       int                     `json:"call_count"`
	Account         AccountInfo             `json:"account"`
	Positions       []PositionInfo          `json:"positions"`
	CandidateCoins  []CandidateCoin         `json:"candidate_coins"`
	MarketDataMap   map[string]*market.Data `json:"-"` // 不序列化，但内部使用
	OITopDataMap    map[string]*OITopData   `json:"-"` // OI Top数据映射
	Performance     interface{}             `json:"-"` // 历史表现分析（logger.PerformanceAnalysis）
	BTCETHLeverage  int                     `json:"-"` // BTC/ETH杠杆倍数（从配置读取）
	AltcoinLeverage int                     `json:"-"` // 山寨币杠杆倍数（从配置读取）
}

// Decision AI的交易决策
type Decision struct {
	// === 基础字段（所有action都需要）===
	Symbol    string `json:"symbol"`
	Action    string `json:"action"` // "open_long", "open_short", "close_long", "close_short", "hold", "wait"
	Reasoning string `json:"reasoning"`

	// === 开仓专用字段（open_long/open_short时必填）===
	Leverage        int     `json:"leverage,omitempty"`          // 杠杆倍数
	PositionSizeUSD float64 `json:"position_size_usd,omitempty"` // 仓位大小（美元）
	StopLoss        float64 `json:"stop_loss,omitempty"`         // 止损价格（具体数值）

	TakeProfitLevels []float64 `json:"take_profit_levels,omitempty"` // 止盈价格数组 [目标1, 目标2, 目标3]
	TrailingStopPct  *float64  `json:"trailing_stop_pct,omitempty"`  // 追踪止损百分比
	ChecklistPassed  *int      `json:"checklist_passed,omitempty"`   // 通过的检查清单条数
	RiskRewardRatio  *float64  `json:"risk_reward_ratio,omitempty"`  // 风险回报比
	SignalType       string    `json:"signal_type,omitempty"`        // 信号类型
	OISignal         string    `json:"oi_signal,omitempty"`          // 持仓量信号状态
	OIAdjustment     string    `json:"oi_adjustment,omitempty"`      // 仓位调整说明

	// === update_stop专用字段 ===
	NewStopLoss *float64 `json:"new_stop_loss,omitempty"` // 新的止损价格

	// === partial_close专用字段 ===
	ClosePercentage *int `json:"close_percentage,omitempty"` // 平仓百分比（30表示30%）
}

// FullDecision AI的完整决策（包含思维链）
type FullDecision struct {
	UserPrompt string     `json:"user_prompt"` // 发送给AI的输入prompt
	CoTTrace   string     `json:"cot_trace"`   // 思维链分析（AI输出）
	Decisions  []Decision `json:"decisions"`   // 具体决策列表
	Timestamp  time.Time  `json:"timestamp"`
}

// GetFullDecision 获取AI的完整交易决策（批量分析所有币种和持仓）
func GetFullDecision(ctx *Context, mcpClient *mcp.Client) (*FullDecision, error) {
	// 1. 为所有币种获取市场数据
	if err := fetchMarketDataForContext(ctx); err != nil {
		return nil, fmt.Errorf("获取市场数据失败: %w", err)
	}

	// 2. 构建 System Prompt（固定规则）和 User Prompt（动态数据）
	systemPrompt := buildSystemPrompt(ctx.Account.TotalEquity, ctx.BTCETHLeverage, ctx.AltcoinLeverage)
	userPrompt := buildUserPrompt(ctx)

	// 3. 调用AI API（使用 system + user prompt）
	aiResponse, err := mcpClient.CallWithMessages(systemPrompt, userPrompt)
	if err != nil {
		return nil, fmt.Errorf("调用AI API失败: %w", err)
	}

	// 4. 解析AI响应
	decision, err := parseFullDecisionResponse(aiResponse, ctx.Account.TotalEquity, ctx.BTCETHLeverage, ctx.AltcoinLeverage)
	if err != nil {
		return nil, fmt.Errorf("解析AI响应失败: %w", err)
	}

	decision.Timestamp = time.Now()
	decision.UserPrompt = userPrompt // 保存输入prompt
	return decision, nil
}

// fetchMarketDataForContext 为上下文中的所有币种获取市场数据和OI数据
func fetchMarketDataForContext(ctx *Context) error {
	ctx.MarketDataMap = make(map[string]*market.Data)
	ctx.OITopDataMap = make(map[string]*OITopData)

	// 收集所有需要获取数据的币种
	symbolSet := make(map[string]bool)

	// 1. 优先获取持仓币种的数据（这是必须的）
	for _, pos := range ctx.Positions {
		symbolSet[pos.Symbol] = true
	}

	// 2. 候选币种数量根据账户状态动态调整
	maxCandidates := calculateMaxCandidates(ctx)
	for i, coin := range ctx.CandidateCoins {
		if i >= maxCandidates {
			break
		}
		symbolSet[coin.Symbol] = true
	}

	// 并发获取市场数据
	// 持仓币种集合（用于判断是否跳过OI检查）
	positionSymbols := make(map[string]bool)
	for _, pos := range ctx.Positions {
		positionSymbols[pos.Symbol] = true
	}

	for symbol := range symbolSet {
		data, err := market.Get(symbol)
		if err != nil {
			// 单个币种失败不影响整体，只记录错误
			continue
		}

		// ⚠️ 流动性过滤：持仓价值低于15M USD的币种不做（多空都不做）
		// 持仓价值 = 持仓量 × 当前价格
		// 但现有持仓必须保留（需要决策是否平仓）
		isExistingPosition := positionSymbols[symbol]
		if !isExistingPosition && data.OpenInterest != nil && data.CurrentPrice > 0 {
			// 计算持仓价值（USD）= 持仓量 × 当前价格
			oiValue := data.OpenInterest.Latest * data.CurrentPrice
			oiValueInMillions := oiValue / 1_000_000 // 转换为百万美元单位
			if oiValueInMillions < 15 {
				log.Printf("⚠️  %s 持仓价值过低(%.2fM USD < 15M)，跳过此币种 [持仓量:%.0f × 价格:%.4f]",
					symbol, oiValueInMillions, data.OpenInterest.Latest, data.CurrentPrice)
				continue
			}
		}

		ctx.MarketDataMap[symbol] = data
	}

	// 加载OI Top数据（不影响主流程）
	oiPositions, err := pool.GetOITopPositions()
	if err == nil {
		for _, pos := range oiPositions {
			// 标准化符号匹配
			symbol := pos.Symbol
			ctx.OITopDataMap[symbol] = &OITopData{
				Rank:              pos.Rank,
				OIDeltaPercent:    pos.OIDeltaPercent,
				OIDeltaValue:      pos.OIDeltaValue,
				PriceDeltaPercent: pos.PriceDeltaPercent,
				NetLong:           pos.NetLong,
				NetShort:          pos.NetShort,
			}
		}
	}

	return nil
}

// calculateMaxCandidates 根据账户状态计算需要分析的候选币种数量
func calculateMaxCandidates(ctx *Context) int {
	// 直接返回候选池的全部币种数量
	// 因为候选池已经在 auto_trader.go 中筛选过了
	// 固定分析前20个评分最高的币种（来自AI500）
	return len(ctx.CandidateCoins)
}

// buildSystemPrompt 构建 System Prompt（固定规则，可缓存）
func buildSystemPrompt(accountEquity float64, btcEthLeverage, altcoinLeverage int) string {
	var sb strings.Builder

	// === 核心使命 ===
	sb.WriteString("你是专业的加密货币交易AI，在币安合约市场进行自主交易。\n\n")
	sb.WriteString("# 🎯 核心目标\n\n")
	sb.WriteString("**最大化夏普比率（Sharpe Ratio）**\n\n")
	sb.WriteString("夏普比率 = 平均收益 / 收益波动率\n\n")
	sb.WriteString("**这意味着**：\n")
	sb.WriteString("- ✅ 高质量交易（高胜率、大盈亏比）→ 提升夏普\n")
	sb.WriteString("- ✅ 稳定收益、控制回撤 → 提升夏普\n")
	sb.WriteString("- ✅ 耐心持仓、让利润奔跑 → 提升夏普\n")
	sb.WriteString("- ❌ 频繁交易、小盈小亏 → 增加波动，严重降低夏普\n")
	sb.WriteString("- ❌ 过度交易、手续费损耗 → 直接亏损\n")
	sb.WriteString("- ❌ 过早平仓、频繁进出 → 错失大行情\n\n")
	sb.WriteString("**关键认知**: 系统每3分钟扫描一次，但主决策基于15分钟K线！\n")
	sb.WriteString("3分钟扫描仅用于动态调整止损，不是开仓信号。\n")
	sb.WriteString("大多数时候应该是 `wait` 或 `hold`，只在极佳机会时才开仓。\n\n")

	// === 硬约束（风险控制）===
	sb.WriteString("# ⚖️ 硬约束（风险控制）\n\n")
	sb.WriteString("1. **风险回报比**: 必须 ≥ 1:2（冒1%风险，赚2%+收益）\n")
	sb.WriteString("2. **最多持仓**: 3个币种（质量>数量）\n")
	sb.WriteString(fmt.Sprintf("3. **单币仓位**: \n"))
	sb.WriteString(fmt.Sprintf("   - 山寨币：%.0f-%.0f U (%dx杠杆) = 单笔5-10%%账户\n",
		accountEquity*0.05, accountEquity*0.10, 3))
	sb.WriteString(fmt.Sprintf("   - BTC/ETH：%.0f-%.0f U (%dx杠杆) = 单笔20-30%%账户\n",
		accountEquity*0.20, accountEquity*0.30, 3))
	sb.WriteString("4. **保证金**: 总使用率 ≤ 70%（留30%缓冲）\n")
	sb.WriteString("5. **单笔最大风险**: 不超过账户净值的2%\n\n")

	// === 做空激励 ===
	sb.WriteString("# 📉 做多做空平衡\n\n")
	sb.WriteString("**重要**: 下跌趋势做空的利润 = 上涨趋势做多的利润\n\n")
	sb.WriteString("- 上涨趋势 → 做多\n")
	sb.WriteString("- 下跌趋势 → 做空\n")
	sb.WriteString("- 震荡市场 → 可交易（降低仓位）\n\n")
	sb.WriteString("**不要有做多偏见！做空是你的核心工具之一**\n\n")

	// === 持仓量背离分析（重新定位）===
	sb.WriteString("# 🔍 持仓量背离分析（辅助工具+关键场景必须）\n\n")
	sb.WriteString("**定位说明**：持仓量(OI)分析是高级辅助工具，在特定场景下是必须条件，在一般场景下用于优化决策。\n\n")

	sb.WriteString("## 📍 使用原则\n\n")
	sb.WriteString("**必须等待持仓量信号的场景**（强制要求）：\n")
	sb.WriteString("1. **抄底交易**：价格大跌>3%后想做多，必须等待「价格↓+空头OI↓」信号\n")
	sb.WriteString("   - 原因：确认多头止损已完成，避免接飞刀\n")
	sb.WriteString("   - 如果是「价格↓+空头OI↑」→ 严禁抄底，空头正在加仓\n\n")

	sb.WriteString("**可选使用的场景**（优化工具）：\n")
	sb.WriteString("2. **仓位优化**：根据持仓量信号调整开仓仓位大小\n")
	sb.WriteString("3. **止盈判断**：持仓中出现反向背离信号，考虑提前止盈\n")
	sb.WriteString("4. **趋势确认**：持仓量配合价格，增强信号可靠性\n\n")

	sb.WriteString("**无信号时**：\n")
	sb.WriteString("- 如果持仓量数据不明显或不可靠，基于技术分析正常交易\n")
	sb.WriteString("- 不要因为没有持仓量信号就不敢交易（抄底除外）\n\n")

	sb.WriteString("## 🔴 做空相关信号\n\n")
	sb.WriteString("**做空信号（空头加仓）**\n")
	sb.WriteString("**价格↓ + 空头OI↑ + 负费率** → 空头主力正在加仓\n")
	sb.WriteString("识别标准（满足2条）：\n")
	sb.WriteString("- □ 价格1小时跌幅 > 2%\n")
	sb.WriteString("- □ 空头OI增长 > 5%\n")
	sb.WriteString("- □ 资金费率 < -0.01%\n")
	sb.WriteString("- □ 成交量放大 > 1.2倍\n")
	sb.WriteString("- □ MACD死叉或EMA20空头排列\n")
	sb.WriteString("**操作**: 可顺势做空，仓位可增加30%\n")
	sb.WriteString("**警告**: 此时严禁抄底做多！\n\n")

	sb.WriteString("**做多陷阱（顶部信号）**\n")
	sb.WriteString("**价格↑ + 多头OI↑ + 高正费率** → 散户FOMO，主力准备收割\n")
	sb.WriteString("识别标准（满足2条）：\n")
	sb.WriteString("- □ 价格1小时涨幅 > 3%\n")
	sb.WriteString("- □ 多头OI增长 > 5%\n")
	sb.WriteString("- □ 资金费率 > 0.05%\n")
	sb.WriteString("- □ K线上影线变长\n")
	sb.WriteString("- □ RSI > 70\n")
	sb.WriteString("**操作**: 如有多单分批止盈，可准备做空\n")
	sb.WriteString("**警告**: 禁止追涨做多！\n\n")

	sb.WriteString("## 🟢 做多相关信号\n\n")
	sb.WriteString("**抄底信号（必须等待！）**\n")
	sb.WriteString("**价格↓ + 空头OI↓** → 多头止损完成，可以抄底\n")
	sb.WriteString("识别标准（满足2条）：\n")
	sb.WriteString("- □ 价格跌幅 > 3%\n")
	sb.WriteString("- □ 空头OI减少 > 3%\n")
	sb.WriteString("- □ 成交量萎缩\n")
	sb.WriteString("- □ RSI < 30\n")
	sb.WriteString("- □ 接近关键支撑位\n")
	sb.WriteString("**操作**: 可小仓位抄底（≤5%账户），设紧密止损\n")
	sb.WriteString("**重要**: 只有出现这个信号才能抄底，否则禁止！\n\n")

	sb.WriteString("**追涨信号（空头挤压）**\n")
	sb.WriteString("**价格↑ + 空头OI↓** → 空头被迫平仓\n")
	sb.WriteString("识别标准（满足2条）：\n")
	sb.WriteString("- □ 价格涨幅 > 2%\n")
	sb.WriteString("- □ 空头OI减少 > 3%\n")
	sb.WriteString("- □ 突破关键阻力位\n")
	sb.WriteString("- □ 成交量持续放大\n")
	sb.WriteString("- □ MACD金叉或EMA20多头排列\n")
	sb.WriteString("**操作**: 可顺势做多，仓位可增加30%\n\n")

	sb.WriteString("## 📊 仓位优化规则\n\n")
	sb.WriteString("**标准仓位**（无明显持仓量信号）：\n")
	sb.WriteString("- 山寨币：5-10%账户\n")
	sb.WriteString("- BTC/ETH：20-30%账户\n\n")

	sb.WriteString("**增强仓位**（有利持仓量信号）：\n")
	sb.WriteString("- 标准仓位 × 1.3（增加30%）\n")
	sb.WriteString("- 例：ETH标准200U → 有空头挤压信号 → 260U\n\n")

	sb.WriteString("**降低仓位**（不利持仓量信号或震荡市）：\n")
	sb.WriteString("- 标准仓位 × 0.5（减少50%）\n")
	sb.WriteString("- 例：山寨标准100U → 有FOMO信号 → 50U\n\n")

	sb.WriteString("**核心认知**：\n")
	sb.WriteString("- 价格+OI同向 = 主力加仓 = 危险信号\n")
	sb.WriteString("- 价格+OI反向 = 清洗完成 = 机会信号\n")
	sb.WriteString("- 持仓量分析的有效周期：1小时-4小时\n\n")

	// === 市场状态识别 ===
	sb.WriteString("# 🌍 市场状态识别\n\n")
	sb.WriteString("在开仓前，必须先判断市场状态，不同状态使用不同策略。\n\n")

	sb.WriteString("## 趋势市（最佳交易环境）\n")
	sb.WriteString("识别标准：\n")
	sb.WriteString("- ADX(14) > 25（趋势强度）\n")
	sb.WriteString("- 价格沿EMA20明确运行（多头：价格在EMA20上方；空头：价格在EMA20下方）\n")
	sb.WriteString("- 连续3根以上同向K线\n")
	sb.WriteString("**策略**: 顺势交易，标准仓位，持仓时间可延长至2-4小时\n\n")

	sb.WriteString("## 震荡市（谨慎环境）\n")
	sb.WriteString("识别标准：\n")
	sb.WriteString("- ADX(14) < 20（无明确趋势）\n")
	sb.WriteString("- 价格在支撑阻力区间来回波动\n")
	sb.WriteString("- 日波动率 < 过去20日均值的0.8倍\n")
	sb.WriteString("**策略**: \n")
	sb.WriteString("  ⚠️ 可以交易，但降低仓位至标准的50%\n")
	sb.WriteString("  ✅ 止损必须更紧密（2-3%以内）\n")
	sb.WriteString("  ✅ 持仓时间缩短（30分钟-1小时）\n")
	sb.WriteString("  ✅ 优先在区间边界操作（支撑做多，阻力做空）\n\n")

	sb.WriteString("## 恐慌市（极高风险）\n")
	sb.WriteString("识别标准：\n")
	sb.WriteString("- BTC日跌幅 > 5%\n")
	sb.WriteString("- 恐慌贪婪指数 < 25\n")
	sb.WriteString("- 全市场暴跌（80%币种下跌）\n")
	sb.WriteString("**策略**: \n")
	sb.WriteString("  ⚠️ 可以做空，但仓位减半\n")
	sb.WriteString("  ❌ 严禁抄底（除非出现明确的持仓量抄底信号）\n")
	sb.WriteString("  ✅ 优先保护资金，降低所有仓位\n\n")

	// === 交易频率认知 ===
	sb.WriteString("# ⏱️ 交易频率认知\n\n")
	sb.WriteString("**量化标准**:\n")
	sb.WriteString("- 优秀交易员：每天2-6笔 = 每小时0.1-0.3笔\n")
	sb.WriteString("- 过度交易：每小时>2笔 = 严重问题\n")
	sb.WriteString("- 最佳节奏：开仓后持有至少1-2小时（短线）或4-8小时（波段）\n\n")
	sb.WriteString("**自查**:\n")
	sb.WriteString("如果你发现自己每个周期都在交易 → 说明标准太低\n")
	sb.WriteString("如果你发现持仓<1小时就平仓 → 说明太急躁\n\n")

	// === 开仓标准 ===
	sb.WriteString("# 🎯 开仓标准（基于技术分析）\n\n")
	sb.WriteString("只在**强信号**时开仓，不确定就观望。\n\n")
	sb.WriteString("**你拥有的完整数据**：\n")
	sb.WriteString("- 📊 **价格序列**：3分钟序列(实时监控) + 15分钟K线(主决策) + 4小时K线(趋势)\n")
	sb.WriteString("- 📈 **技术指标序列**：EMA20、MACD、RSI7、RSI14、ADX\n")
	sb.WriteString("- 💰 **资金序列**：成交量、持仓量(OI)、多空OI比、资金费率\n")
	sb.WriteString("- 🎯 **筛选标记**：AI500评分 / OI_Top排名（如有）\n\n")

	sb.WriteString("**开仓前必须通过的检查清单（至少满足2条）**：\n")
	sb.WriteString("1. □ **趋势确认**：ADX>20 且 价格沿EMA20运行，或连续2-3根同向K线\n")
	sb.WriteString("2. □ **动量指标**：RSI突破50中线 或 MACD金叉/死叉\n")
	sb.WriteString("3. □ **成交量配合**：放量突破（Vol > 20日均量1.3倍）\n")
	sb.WriteString("4. □ **关键位突破**：突破/跌破支撑阻力位、前高前低、整数关口\n")
	sb.WriteString("5. □ **风险回报比**：≥1:2（止损距离 vs 止盈距离）\n")
	sb.WriteString("6. □ **市场环境**：非恐慌市（震荡市可降低仓位交易）\n")
	sb.WriteString("7. □ **资金费率配合**：做多时费率不超过0.05%，做空时费率为负更好\n\n")

	sb.WriteString("**特殊规则**：\n")
	sb.WriteString("- **抄底操作**（价格大跌>3%后做多）：必须额外满足持仓量抄底信号\n")
	sb.WriteString("- **追高操作**（价格大涨>3%后追涨）：需确认无「价格↑+多头OI↑」FOMO信号\n\n")

	sb.WriteString("**严格禁止的低质量信号**：\n")
	sb.WriteString("- ❌ 单一维度（只看一个指标）\n")
	sb.WriteString("- ❌ 相互矛盾（价格涨但量萎缩、OI不配合）\n")
	sb.WriteString("- ❌ 刚平仓不久（<30分钟，避免情绪化报复交易）\n")
	sb.WriteString("- ❌ 刚止损后立即反向开仓（需冷静至少15分钟）\n")
	sb.WriteString("- ❌ 在明确的持仓量危险信号下逆势交易\n\n")

	// === 动态止损机制 ===
	sb.WriteString("# 🛡️ 动态止损止盈机制\n\n")
	sb.WriteString("开仓时必须设置：初始止损 + 追踪止损 + 分批止盈\n\n")

	sb.WriteString("## 初始止损\n")
	sb.WriteString("- **做多止损**：设在近期关键低点下方2%，或EMA20下方3%\n")
	sb.WriteString("- **做空止损**：设在近期关键高点上方2%，或EMA20上方3%\n")
	sb.WriteString("- **最大止损**：不超过开仓价的5%（山寨币可放宽到7%）\n\n")

	sb.WriteString("## 追踪止损（让利润奔跑）\n")
	sb.WriteString("当浮盈达到3%后，启动追踪止损：\n")
	sb.WriteString("- 浮盈3-5%：止损移至开仓价（保本）\n")
	sb.WriteString("- 浮盈5-10%：止损跟随价格，保持5%距离\n")
	sb.WriteString("- 浮盈>10%：止损跟随价格，保持7%距离\n\n")

	sb.WriteString("## 分批止盈（锁定利润）\n")
	sb.WriteString("达到目标后分批卖出，降低回撤风险：\n")
	sb.WriteString("- 第一目标（+3%）：平仓30%，剩余持仓移止损至保本\n")
	sb.WriteString("- 第二目标（+6%）：平仓30%，剩余持仓追踪止损\n")
	sb.WriteString("- 第三目标（+9%或更高）：平仓剩余40%\n\n")

	sb.WriteString("## 持仓量预警止盈（新增）\n")
	sb.WriteString("持仓过程中，如果出现反向持仓量信号，考虑提前止盈：\n")
	sb.WriteString("- **持有多单**：观察到「价格↑+多头OI↑>8%+费率>0.08%」→ 散户FOMO，考虑止盈\n")
	sb.WriteString("- **持有空单**：观察到「价格↓+空头OI↓>5%」→ 空头平仓，考虑止盈\n\n")

	sb.WriteString("## 时间止损\n")
	sb.WriteString("如果持仓超过以下时间未触发止盈/止损，需重新评估：\n")
	sb.WriteString("- 短线单：超过4小时，如无进展考虑离场\n")
	sb.WriteString("- 波段单：超过24小时无进展，评估是否继续持有\n\n")

	// === 夏普比率自我进化 ===
	sb.WriteString("# 🧬 夏普比率自我进化\n\n")
	sb.WriteString("每次你会收到**夏普比率**作为绩效反馈（基于24小时滚动窗口计算）：\n\n")
	sb.WriteString("**夏普比率 < -0.5** (持续亏损):\n")
	sb.WriteString("  → 🛑 立即停止交易，连续观望至少6个周期（18分钟）\n")
	sb.WriteString("  → 🔍 深度反思：\n")
	sb.WriteString("     • 交易频率过高？（每小时>2次就是过度）\n")
	sb.WriteString("     • 持仓时间过短？（<1小时就是过早平仓）\n")
	sb.WriteString("     • 是否在危险持仓量信号下逆势交易？\n")
	sb.WriteString("     • 是否在震荡市用了过大仓位？\n")
	sb.WriteString("     • 是否忽视做空机会？（单边做多是错误的）\n")
	sb.WriteString("     • 开仓检查清单是否满足<2条？\n\n")
	sb.WriteString("**夏普比率 -0.5 ~ 0** (轻微亏损):\n")
	sb.WriteString("  → ⚠️ 严格控制：只做满足≥3条检查清单的交易\n")
	sb.WriteString("  → 减少交易频率：每小时最多1笔新开仓\n")
	sb.WriteString("  → 耐心持仓：至少持有1小时以上\n")
	sb.WriteString("  → 优先使用持仓量信号优化决策\n\n")
	sb.WriteString("**夏普比率 0 ~ 0.7** (正收益):\n")
	sb.WriteString("  → ✅ 维持当前策略\n")
	sb.WriteString("  → 记录并复盘成功交易的共同特征\n\n")
	sb.WriteString("**夏普比率 > 0.7** (优异表现):\n")
	sb.WriteString("  → 🚀 可适度扩大仓位（单笔最多增加30%）\n")
	sb.WriteString("  → 继续保持纪律，避免过度自信\n\n")
	sb.WriteString("**关键**: 夏普比率是唯一指标，它会自然惩罚频繁交易和过度进出。\n\n")

	// === 风控熔断机制 ===
	sb.WriteString("# 🚨 风控熔断机制（自动触发）\n\n")
	sb.WriteString("以下情况触发自动熔断，优先级高于一切交易信号：\n\n")

	sb.WriteString("**一级熔断（立即停止所有交易）**：\n")
	sb.WriteString("1. 单日亏损 > 5% 账户净值 → 停止交易至次日\n")
	sb.WriteString("2. 连续3笔止损 → 暂停交易1小时，强制冷静\n")
	sb.WriteString("3. 保证金使用率 > 70% → 禁止开新仓，只允许平仓\n\n")

	sb.WriteString("**二级熔断（限制交易）**：\n")
	sb.WriteString("1. 单日亏损 2-5% → 仅允许防守性操作（止损/止盈）\n")
	sb.WriteString("2. 连续2笔止损 → 下一笔必须满足≥3条检查清单\n")
	sb.WriteString("3. 持仓超过2个币种 → 禁止开第3个，优先平仓弱势币种\n\n")

	sb.WriteString("**三级警告（提高标准）**：\n")
	sb.WriteString("1. 单日亏损 1-2% → 只做高确定性交易（≥3条检查清单）\n")
	sb.WriteString("2. 最近1小时开仓>2笔 → 强制观望30分钟\n\n")

	// === 决策流程 ===
	sb.WriteString("# 📋 决策流程（每个周期必须执行）\n\n")
	sb.WriteString("**第一步：风控检查**\n")
	sb.WriteString("- 是否触发熔断机制？如是，立即执行熔断指令\n")
	sb.WriteString("- 保证金使用率多少？是否接近警戒线？\n\n")

	sb.WriteString("**第二步：夏普比率分析**\n")
	sb.WriteString("- 当前24小时夏普比率多少？\n")
	sb.WriteString("- 策略是否有效？需要调整吗？\n")
	sb.WriteString("- 如果<-0.5，立即停止交易\n\n")

	sb.WriteString("**第三步：市场状态识别**\n")
	sb.WriteString("- 当前是趋势市、震荡市还是恐慌市？\n")
	sb.WriteString("- 如果是恐慌市，极度谨慎；震荡市降低仓位\n\n")

	sb.WriteString("**第四步：评估现有持仓（如有）**\n")
	sb.WriteString("- 趋势是否改变？持仓量是否出现反向背离信号？\n")
	sb.WriteString("- 是否触发追踪止损？是否达到分批止盈点位？\n")
	sb.WriteString("- 持仓时间多久？是否需要时间止损？\n")
	sb.WriteString("- 如有持仓量预警信号，考虑提前止盈\n\n")

	sb.WriteString("**第五步：寻找新机会**\n")
	sb.WriteString("- 扫描所有币种的技术指标\n")
	sb.WriteString("- 开仓检查清单能满足几条？（必须≥2条）\n")
	sb.WriteString("- 风险回报比是否≥1:2？\n")
	sb.WriteString("- **特别检查**：\n")
	sb.WriteString("  * 如果是抄底：必须有持仓量抄底信号\n")
	sb.WriteString("  * 如果有持仓量有利信号：可增加仓位30%\n")
	sb.WriteString("  * 如果有持仓量危险信号：降低仓位50%或观望\n")
	sb.WriteString("- 多空机会都要考虑，不要偏见\n\n")

	sb.WriteString("**第六步：计算价格**\n")
	sb.WriteString("- 做多：止损=当前价×0.97，止盈=[当前价×1.03, ×1.06, ×1.09]\n")
	sb.WriteString("- 做空：止损=当前价×1.03，止盈=[当前价×0.97, ×0.94, ×0.91]\n\n")

	sb.WriteString("**第七步：输出决策**\n")
	sb.WriteString("思维链分析 + JSON数组\n\n")

	// === 输出格式（强化止损止盈计算） ===
	sb.WriteString("# 📤 输出格式（严格遵守）\n\n")

	sb.WriteString("**思维链（简洁分析）**\n")
	sb.WriteString("```\n")
	sb.WriteString("【风控】保证金45%正常\n")
	sb.WriteString("【夏普】0.3有效\n")
	sb.WriteString("【市场】BTC上升趋势\n")
	sb.WriteString("【机会】ETHUSDT当前3850，突破+金叉+量能，满足3条\n")
	sb.WriteString("【价格】止损3735，止盈3966/4081/4197\n")
	sb.WriteString("【决策】开多130U，3x杠杆\n")
	sb.WriteString("```\n\n")

	sb.WriteString("**JSON决策数组（所有价格必须是纯数字）**\n\n")
	sb.WriteString("⚠️ **重要**: JSON必须输出完整的决策对象，不是单纯的数字数组！\n\n")
	sb.WriteString("❌ 错误: `[8.20, 8.50, 8.80]` ← 只有数字，缺少symbol、action等字段\n")
	sb.WriteString("✅ 正确: `[{\"symbol\": \"...\", \"action\": \"...\", \"stop_loss\": ..., \"take_profit_levels\": [8.20, 8.50, 8.80], ...}]`\n\n")

	sb.WriteString("```json\n[\n")
	sb.WriteString("  {\n")
	sb.WriteString("    \"symbol\": \"ETHUSDT\",\n")
	sb.WriteString("    \"action\": \"open_long\",\n")
	sb.WriteString(fmt.Sprintf("    \"leverage\": %d,\n", 3))
	sb.WriteString(fmt.Sprintf("    \"position_size_usd\": %.0f,\n", accountEquity*0.10))
	sb.WriteString("    \"stop_loss\": 3735,\n")
	sb.WriteString("    \"take_profit_levels\": [3966, 4081, 4197],\n")
	sb.WriteString("    \"trailing_stop_pct\": 5,\n")
	sb.WriteString("    \"checklist_passed\": 3,\n")
	sb.WriteString("    \"risk_reward_ratio\": 2.5,\n")
	sb.WriteString("    \"signal_type\": \"空头挤压\",\n")
	sb.WriteString("    \"oi_signal\": \"空头OI降8%\",\n")
	sb.WriteString("    \"oi_adjustment\": \"+30%\",\n")
	sb.WriteString("    \"reasoning\": \"突破阻力+金叉+量1.5倍\"\n")
	sb.WriteString("  },\n")
	sb.WriteString("  {\n")
	sb.WriteString("    \"action\": \"wait\",\n")
	sb.WriteString("    \"reasoning\": \"其他币种无明确机会\"\n")
	sb.WriteString("  }\n")
	sb.WriteString("]\n```\n\n")

	sb.WriteString("**输出要点**:\n")
	sb.WriteString("1. 最外层是数组 `[]`\n")
	sb.WriteString("2. 数组里必须是完整的对象 `{symbol, action, leverage, ...}`，不能只是数字\n")
	sb.WriteString("3. `take_profit_levels` 是对象中的一个字段，包含3个数字\n\n")

	sb.WriteString("**字段说明**:\n\n")
	sb.WriteString("**action类型**: open_long | open_short | close_long | close_short | update_stop | partial_close | hold | wait\n\n")

	sb.WriteString("**开仓必填**: symbol, action, leverage, position_size_usd, stop_loss, take_profit_levels(3个数字的数组), trailing_stop_pct, checklist_passed(≥2), risk_reward_ratio(≥2), signal_type, oi_signal, oi_adjustment, reasoning\n\n")

	sb.WriteString("**其他操作**:\n")
	sb.WriteString("- close_*: symbol, action, reasoning\n")
	sb.WriteString("- update_stop: symbol, action, new_stop_loss, reasoning\n")
	sb.WriteString("- partial_close: symbol, action, close_percentage, reasoning\n")
	sb.WriteString("- hold/wait: action, reasoning\n\n")

	// === 关键提醒 ===
	sb.WriteString("---\n\n")
	sb.WriteString("# 🎓 核心交易理念（必须内化）\n\n")
	sb.WriteString("1. **技术分析是开仓基础** - 满足≥2条检查清单即可交易\n")
	sb.WriteString("2. **持仓量分析是辅助工具** - 用于优化仓位和确认特殊场景（抄底必须）\n")
	sb.WriteString("3. **目标是夏普比率，不是交易频率** - 宁可错过，不做低质量交易\n")
	sb.WriteString("4. **做空 = 做多** - 都是赚钱工具，不要有偏见\n")
	sb.WriteString("5. **风险回报比1:2是底线** - 没有例外\n")
	sb.WriteString("6. **让利润奔跑，快速止损** - 追踪止损和分批止盈是必须的\n")
	sb.WriteString("7. **市场状态识别** - 恐慌市谨慎，震荡市降低仓位\n")
	sb.WriteString("8. **熔断机制优先级最高** - 保护本金比赚钱更重要\n")
	sb.WriteString("9. **3分钟扫描≠3分钟决策** - 主决策基于15分钟K线\n")
	sb.WriteString("10. **情绪是敌人** - 刚止损后必须冷静15分钟再考虑新仓\n")
	sb.WriteString("11. **止损止盈必须是具体价格** - 输出JSON前必须先计算好，不能是0\n")
	sb.WriteString("12. **每笔交易都必须经得起复盘检验** - 记录完整的决策理由和计算过程\n\n")

	sb.WriteString("**记住**: 你是专业交易AI，不是赌徒。每一笔交易都必须有清晰的逻辑、完整的数据支持、明确的止损止盈计划。持仓量分析让你的交易更优秀，但不是交易的前提条件（抄底除外）。在思维链中展示价格计算过程，在JSON中输出计算好的具体数值。止损止盈不能是0或负数！\n\n")
	sb.WriteString("现在开始执行交易决策。\n")

	return sb.String()
}

// buildUserPrompt 构建 User Prompt（动态数据）
func buildUserPrompt(ctx *Context) string {
	var sb strings.Builder

	// 系统状态
	sb.WriteString(fmt.Sprintf("**时间**: %s | **周期**: #%d | **运行**: %d分钟\n\n",
		ctx.CurrentTime, ctx.CallCount, ctx.RuntimeMinutes))

	// BTC 市场
	if btcData, hasBTC := ctx.MarketDataMap["BTCUSDT"]; hasBTC {
		sb.WriteString(fmt.Sprintf("**BTC**: %.2f (1h: %+.2f%%, 4h: %+.2f%%) | MACD: %.4f | RSI: %.2f\n\n",
			btcData.CurrentPrice, btcData.PriceChange1h, btcData.PriceChange4h,
			btcData.CurrentMACD, btcData.CurrentRSI7))
	}

	// 账户
	sb.WriteString(fmt.Sprintf("**账户**: 净值%.2f | 余额%.2f (%.1f%%) | 盈亏%+.2f%% | 保证金%.1f%% | 持仓%d个\n\n",
		ctx.Account.TotalEquity,
		ctx.Account.AvailableBalance,
		(ctx.Account.AvailableBalance/ctx.Account.TotalEquity)*100,
		ctx.Account.TotalPnLPct,
		ctx.Account.MarginUsedPct,
		ctx.Account.PositionCount))

	// 持仓（完整市场数据）
	if len(ctx.Positions) > 0 {
		sb.WriteString("## 当前持仓\n")
		for i, pos := range ctx.Positions {
			// 计算持仓时长
			holdingDuration := ""
			if pos.UpdateTime > 0 {
				durationMs := time.Now().UnixMilli() - pos.UpdateTime
				durationMin := durationMs / (1000 * 60) // 转换为分钟
				if durationMin < 60 {
					holdingDuration = fmt.Sprintf(" | 持仓时长%d分钟", durationMin)
				} else {
					durationHour := durationMin / 60
					durationMinRemainder := durationMin % 60
					holdingDuration = fmt.Sprintf(" | 持仓时长%d小时%d分钟", durationHour, durationMinRemainder)
				}
			}

			sb.WriteString(fmt.Sprintf("%d. %s %s | 入场价%.4f 当前价%.4f | 盈亏%+.2f%% | 杠杆%dx | 保证金%.0f | 强平价%.4f%s\n\n",
				i+1, pos.Symbol, strings.ToUpper(pos.Side),
				pos.EntryPrice, pos.MarkPrice, pos.UnrealizedPnLPct,
				pos.Leverage, pos.MarginUsed, pos.LiquidationPrice, holdingDuration))

			// 使用FormatMarketData输出完整市场数据
			if marketData, ok := ctx.MarketDataMap[pos.Symbol]; ok {
				sb.WriteString(market.Format(marketData))
				sb.WriteString("\n")
			}
		}
	} else {
		sb.WriteString("**当前持仓**: 无\n\n")
	}

	// 候选币种（完整市场数据）
	sb.WriteString(fmt.Sprintf("## 候选币种 (%d个)\n\n", len(ctx.MarketDataMap)))
	displayedCount := 0
	for _, coin := range ctx.CandidateCoins {
		marketData, hasData := ctx.MarketDataMap[coin.Symbol]
		if !hasData {
			continue
		}
		displayedCount++

		sourceTags := ""
		if len(coin.Sources) > 1 {
			sourceTags = " (AI500+OI_Top双重信号)"
		} else if len(coin.Sources) == 1 && coin.Sources[0] == "oi_top" {
			sourceTags = " (OI_Top持仓增长)"
		}

		// 使用FormatMarketData输出完整市场数据
		sb.WriteString(fmt.Sprintf("### %d. %s%s\n\n", displayedCount, coin.Symbol, sourceTags))
		sb.WriteString(market.Format(marketData))
		sb.WriteString("\n")
	}
	sb.WriteString("\n")

	// 夏普比率（直接传值，不要复杂格式化）
	if ctx.Performance != nil {
		// 直接从interface{}中提取SharpeRatio
		type PerformanceData struct {
			SharpeRatio float64 `json:"sharpe_ratio"`
		}
		var perfData PerformanceData
		if jsonData, err := json.Marshal(ctx.Performance); err == nil {
			if err := json.Unmarshal(jsonData, &perfData); err == nil {
				sb.WriteString(fmt.Sprintf("## 📊 夏普比率: %.2f\n\n", perfData.SharpeRatio))
			}
		}
	}

	sb.WriteString("---\n\n")
	sb.WriteString("现在请分析并输出决策（思维链 + JSON）\n")

	return sb.String()
}

// parseFullDecisionResponse 解析AI的完整决策响应
func parseFullDecisionResponse(aiResponse string, accountEquity float64, btcEthLeverage, altcoinLeverage int) (*FullDecision, error) {
	// 1. 提取思维链
	cotTrace := extractCoTTrace(aiResponse)

	// 2. 提取JSON决策列表
	decisions, err := extractDecisions(aiResponse)
	if err != nil {
		return &FullDecision{
			CoTTrace:  cotTrace,
			Decisions: []Decision{},
		}, fmt.Errorf("提取决策失败: %w\n\n=== AI思维链分析 ===\n%s", err, cotTrace)
	}

	// 3. 验证决策
	if err := validateDecisions(decisions, accountEquity, btcEthLeverage, altcoinLeverage); err != nil {
		return &FullDecision{
			CoTTrace:  cotTrace,
			Decisions: decisions,
		}, fmt.Errorf("决策验证失败: %w\n\n=== AI思维链分析 ===\n%s", err, cotTrace)
	}

	return &FullDecision{
		CoTTrace:  cotTrace,
		Decisions: decisions,
	}, nil
}

// extractCoTTrace 提取思维链分析
func extractCoTTrace(response string) string {
	// 查找JSON数组的开始位置
	jsonStart := strings.Index(response, "[")

	if jsonStart > 0 {
		// 思维链是JSON数组之前的内容
		return strings.TrimSpace(response[:jsonStart])
	}

	// 如果找不到JSON，整个响应都是思维链
	return strings.TrimSpace(response)
}

// extractDecisions 提取JSON决策列表
func extractDecisions(response string) ([]Decision, error) {
	// 直接查找JSON数组 - 找第一个完整的JSON数组
	arrayStart := strings.Index(response, "[")
	if arrayStart == -1 {
		return nil, fmt.Errorf("无法找到JSON数组起始")
	}

	// 从 [ 开始，匹配括号找到对应的 ]
	arrayEnd := findMatchingBracket(response, arrayStart)
	if arrayEnd == -1 {
		return nil, fmt.Errorf("无法找到JSON数组结束")
	}

	jsonContent := strings.TrimSpace(response[arrayStart : arrayEnd+1])

	// 🔧 修复常见的JSON格式错误：缺少引号的字段值
	// 匹配: "reasoning": 内容"}  或  "reasoning": 内容}  (没有引号)
	// 修复为: "reasoning": "内容"}
	// 使用简单的字符串扫描而不是正则表达式
	jsonContent = fixMissingQuotes(jsonContent)

	// 解析JSON
	var decisions []Decision
	if err := json.Unmarshal([]byte(jsonContent), &decisions); err != nil {
		fmt.Printf("JSON内容: %s\n", jsonContent)
		return nil, fmt.Errorf("JSON解析失败: %w\nJSON内容: %s", err, jsonContent)
	}

	return decisions, nil
}

// fixMissingQuotes 替换中文引号为英文引号（避免输入法自动转换）
func fixMissingQuotes(jsonStr string) string {
	jsonStr = strings.ReplaceAll(jsonStr, "\u201c", "\"") // "
	jsonStr = strings.ReplaceAll(jsonStr, "\u201d", "\"") // "
	jsonStr = strings.ReplaceAll(jsonStr, "\u2018", "'")  // '
	jsonStr = strings.ReplaceAll(jsonStr, "\u2019", "'")  // '
	return jsonStr
}

// validateDecisions 验证所有决策（需要账户信息和杠杆配置）
func validateDecisions(decisions []Decision, accountEquity float64, btcEthLeverage, altcoinLeverage int) error {
	for i, decision := range decisions {
		if err := validateDecision(&decision, accountEquity, btcEthLeverage, altcoinLeverage); err != nil {
			return fmt.Errorf("决策 #%d 验证失败: %w", i+1, err)
		}
	}
	return nil
}

// findMatchingBracket 查找匹配的右括号
func findMatchingBracket(s string, start int) int {
	if start >= len(s) || s[start] != '[' {
		return -1
	}

	depth := 0
	for i := start; i < len(s); i++ {
		switch s[i] {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return i
			}
		}
	}

	return -1
}

// validateDecision 验证单个决策的有效性
func validateDecision(d *Decision, accountEquity float64, btcEthLeverage, altcoinLeverage int) error {
	// 验证action
	validActions := map[string]bool{
		"open_long":   true,
		"open_short":  true,
		"close_long":  true,
		"close_short": true,
		"hold":        true,
		"wait":        true,
	}

	if !validActions[d.Action] {
		return fmt.Errorf("无效的action: %s", d.Action)
	}

	// 开仓操作必须提供完整参数
	if d.Action == "open_long" || d.Action == "open_short" {
		// 根据币种使用配置的杠杆上限
		maxLeverage := altcoinLeverage          // 山寨币使用配置的杠杆
		maxPositionValue := accountEquity * 1.5 // 山寨币最多1.5倍账户净值
		if d.Symbol == "BTCUSDT" || d.Symbol == "ETHUSDT" {
			maxLeverage = btcEthLeverage          // BTC和ETH使用配置的杠杆
			maxPositionValue = accountEquity * 10 // BTC/ETH最多10倍账户净值
		}

		if d.Leverage <= 0 || d.Leverage > maxLeverage {
			return fmt.Errorf("杠杆必须在1-%d之间（%s，当前配置上限%d倍）: %d", maxLeverage, d.Symbol, maxLeverage, d.Leverage)
		}
		if d.PositionSizeUSD <= 0 {
			return fmt.Errorf("仓位大小必须大于0: %.2f", d.PositionSizeUSD)
		}
		// 验证仓位价值上限（加1%容差以避免浮点数精度问题）
		tolerance := maxPositionValue * 0.01 // 1%容差
		if d.PositionSizeUSD > maxPositionValue+tolerance {
			if d.Symbol == "BTCUSDT" || d.Symbol == "ETHUSDT" {
				return fmt.Errorf("BTC/ETH单币种仓位价值不能超过%.0f USDT（10倍账户净值），实际: %.0f", maxPositionValue, d.PositionSizeUSD)
			} else {
				return fmt.Errorf("山寨币单币种仓位价值不能超过%.0f USDT（1.5倍账户净值），实际: %.0f", maxPositionValue, d.PositionSizeUSD)
			}
		}

		// 验证止损
		if d.StopLoss <= 0 {
			return fmt.Errorf("止损必须大于0")
		}

		// 验证止盈数组
		if len(d.TakeProfitLevels) != 3 {
			return fmt.Errorf("止盈必须包含3个目标价格，当前: %d个", len(d.TakeProfitLevels))
		}
		for i, tp := range d.TakeProfitLevels {
			if tp <= 0 {
				return fmt.Errorf("止盈目标%d必须大于0: %.2f", i+1, tp)
			}
		}

		// 验证止损止盈的合理性和顺序
		if d.Action == "open_long" {
			// 做多：止损 < 止盈1 < 止盈2 < 止盈3
			if d.StopLoss >= d.TakeProfitLevels[0] {
				return fmt.Errorf("做多时止损价(%.2f)必须小于第一止盈目标(%.2f)", d.StopLoss, d.TakeProfitLevels[0])
			}
			if d.TakeProfitLevels[0] >= d.TakeProfitLevels[1] || d.TakeProfitLevels[1] >= d.TakeProfitLevels[2] {
				return fmt.Errorf("做多时止盈目标必须递增: [%.2f, %.2f, %.2f]",
					d.TakeProfitLevels[0], d.TakeProfitLevels[1], d.TakeProfitLevels[2])
			}
		} else { // open_short
			// 做空：止盈1 < 止盈2 < 止盈3 < 止损
			if d.StopLoss <= d.TakeProfitLevels[0] {
				return fmt.Errorf("做空时止损价(%.2f)必须大于第一止盈目标(%.2f)", d.StopLoss, d.TakeProfitLevels[0])
			}
			if d.TakeProfitLevels[0] <= d.TakeProfitLevels[1] || d.TakeProfitLevels[1] <= d.TakeProfitLevels[2] {
				return fmt.Errorf("做空时止盈目标必须递减: [%.2f, %.2f, %.2f]",
					d.TakeProfitLevels[0], d.TakeProfitLevels[1], d.TakeProfitLevels[2])
			}
		}

		// 验证风险回报比（使用第一止盈目标计算，必须≥2.0）
		var entryPrice float64
		if d.Action == "open_long" {
			// 做多：入场价假设在止损和第一止盈之间（20%位置）
			entryPrice = d.StopLoss + (d.TakeProfitLevels[0]-d.StopLoss)*0.2
		} else {
			// 做空：入场价假设在第一止盈和止损之间（20%位置）
			entryPrice = d.StopLoss - (d.StopLoss-d.TakeProfitLevels[0])*0.2
		}

		var riskPercent, rewardPercent, riskRewardRatio float64
		if d.Action == "open_long" {
			riskPercent = (entryPrice - d.StopLoss) / entryPrice * 100
			rewardPercent = (d.TakeProfitLevels[0] - entryPrice) / entryPrice * 100
			if riskPercent > 0 {
				riskRewardRatio = rewardPercent / riskPercent
			}
		} else {
			riskPercent = (d.StopLoss - entryPrice) / entryPrice * 100
			rewardPercent = (entryPrice - d.TakeProfitLevels[0]) / entryPrice * 100
			if riskPercent > 0 {
				riskRewardRatio = rewardPercent / riskPercent
			}
		}

		// 硬约束：风险回报比必须≥2.0（使用第一止盈目标）
		if riskRewardRatio < 2.0 {
			return fmt.Errorf("风险回报比过低(%.2f:1)，必须≥2.0:1 [风险:%.2f%% 收益:%.2f%%] [止损:%.2f 止盈1:%.2f]",
				riskRewardRatio, riskPercent, rewardPercent, d.StopLoss, d.TakeProfitLevels[0])
		}
	}

	return nil
}
