from __future__ import annotations

import argparse
from dataclasses import dataclass


BASE_TEMPLATE = """Act as a professional equity research analyst and portfolio strategist.

Produce an institutional-quality investment memo on {subject}.

Use the most recent available information and clearly distinguish:
- Facts
- Reasonable inferences
- Explicit assumptions

Write in plain English with analytical depth, not hype.

Base your analysis on information available as of {as_of_date}.

Required output structure:

1. Investment Summary
- A one-paragraph overview
- A clear rating: Buy, Hold, or Sell
- 3-5 key reasons supporting the rating

2. Business Overview
- How the company makes money
- Main products, services, and business segments
- Geographic exposure
- Customer base and end-market exposure

3. Industry and Competitive Position
- Industry structure and outlook
- Main competitors
- Market position and differentiation
- Long-term positioning
- Whether the company has a sustainable competitive advantage
- Moat assessment across brand, scale, cost advantage, switching costs, IP, network effects, regulation, and distribution

4. Financial Statement Analysis
- Revenue growth trends
- Gross margin, operating margin, and net margin trends
- Profitability quality
- Free cash flow quality
- Balance sheet strength
- Debt, liquidity, and capital allocation
- Return metrics such as ROIC, ROE, or ROA where relevant
- Key strengths
- Key weaknesses
- Warning signs investors should know

5. Growth Drivers
- Near-term catalysts
- Long-term secular tailwinds
- Product and pricing opportunities
- Geographic expansion opportunities
- Margin expansion opportunities
- Capital allocation opportunities
- Management execution priorities

6. Risks
- Industry risks
- Competitive threats
- Economic and macro risks
- Regulatory and legal risks
- Management or governance concerns
- Financial risks
- Valuation risks

7. Valuation
- Assess valuation using appropriate methods
- Compare valuation versus major peers
- Compare valuation versus the company’s historical range
- Estimate intrinsic value using reasonable assumptions for revenue growth, margins, free cash flow, terminal multiple, and discount rate
- Explain every assumption clearly
- Include base, bull, and bear scenarios
- State whether the stock appears undervalued, fairly valued, or overvalued

8. Latest Earnings Review
- The most important takeaways
- Positive surprises
- Negative surprises
- Margin or revenue inflections
- Management commentary
- Guidance changes
- What investors should watch next quarter and over the next 12 months

9. Bull vs. Bear Debate
- Strongest bull arguments
- Strongest bear arguments
- Which side is more compelling today and why

10. Portfolio Fit
Based on:
- Investment goals: {goals}
- Risk tolerance: {risk_tolerance}
- Time horizon: {time_horizon}

Explain:
- Whether this stock deserves a place in the portfolio
- What role it would play: compounder, value play, cyclical, turnaround, income, speculative growth, etc.
- Position sizing considerations
- What type of investor should or should not own it

11. Final Verdict
- Buy, Hold, or Sell
- 12-36 month outlook
- Target investor type
- Expected return profile
- Key conditions that would make you upgrade or downgrade the stock

Additional instructions:
- Be specific and evidence-based
- Do not rely on vague generalities
- If an important data point is missing, say so clearly
- Highlight both upside and downside
- Keep the tone professional and objective
- Avoid generic disclaimers unless absolutely necessary
"""


COMPARISON_APPENDIX = """

Also include a direct comparison between {subject} and {comparison_target} covering:
- Growth
- Profitability
- Valuation
- Financial health
- Risk
- Long-term potential

Conclude which is the better investment opportunity today and why.
"""


EARNINGS_TEMPLATE = """Act as a professional equity research analyst.

Analyze the latest earnings report from {subject}.
Base your analysis on information available as of {as_of_date}.

Cover:
- Headline results versus expectations
- Revenue, margins, EPS, free cash flow, and guidance
- Segment performance
- Positive surprises
- Negative surprises
- Management commentary
- Capital allocation
- Risks and what investors should watch next

End with:
- Is the earnings report thesis-confirming, thesis-neutral, or thesis-breaking?
- What should long-term investors do next?
"""


VALUATION_TEMPLATE = """Act as a professional equity research analyst focused on valuation.

Estimate the intrinsic value of {subject} as of {as_of_date}.

Use reasonable assumptions for:
- Revenue growth
- Operating margins
- Taxes
- Reinvestment needs
- Free cash flow conversion
- Discount rate
- Terminal growth or exit multiple

Requirements:
- Show base, bull, and bear cases
- Explain every assumption clearly
- Compare your result to the current market valuation
- State the implied upside or downside
- Explain the most important variables driving valuation sensitivity
"""


DEBATE_TEMPLATE = """Act as two investors debating {subject} as of {as_of_date}.

Create a balanced bull vs. bear debate with:
- The strongest bull case
- The strongest bear case
- Rebuttals from each side
- The key facts that matter most

End with:
- Which side is more compelling today
- What evidence would change the conclusion
"""


@dataclass(frozen=True)
class ResearchPromptSpec:
    subject: str
    goals: str = "long-term capital appreciation"
    risk_tolerance: str = "moderate"
    time_horizon: str = "5+ years"
    as_of_date: str = "today"
    comparison_target: str = ""


def build_investment_memo_prompt(spec: ResearchPromptSpec) -> str:
    prompt = BASE_TEMPLATE.format(
        subject=spec.subject,
        goals=spec.goals,
        risk_tolerance=spec.risk_tolerance,
        time_horizon=spec.time_horizon,
        as_of_date=spec.as_of_date,
    ).strip()
    if spec.comparison_target.strip():
        prompt += COMPARISON_APPENDIX.format(
            subject=spec.subject,
            comparison_target=spec.comparison_target.strip(),
        )
    return prompt


def build_earnings_prompt(spec: ResearchPromptSpec) -> str:
    return EARNINGS_TEMPLATE.format(
        subject=spec.subject,
        as_of_date=spec.as_of_date,
    ).strip()


def build_valuation_prompt(spec: ResearchPromptSpec) -> str:
    return VALUATION_TEMPLATE.format(
        subject=spec.subject,
        as_of_date=spec.as_of_date,
    ).strip()


def build_debate_prompt(spec: ResearchPromptSpec) -> str:
    return DEBATE_TEMPLATE.format(
        subject=spec.subject,
        as_of_date=spec.as_of_date,
    ).strip()


def build_prompt_for_mode(mode: str, spec: ResearchPromptSpec) -> str:
    mode_key = str(mode or "memo").strip().lower()
    if mode_key == "earnings":
        return build_earnings_prompt(spec)
    if mode_key == "valuation":
        return build_valuation_prompt(spec)
    if mode_key == "debate":
        return build_debate_prompt(spec)
    return build_investment_memo_prompt(spec)


def build_research_packet(
    *,
    spec: ResearchPromptSpec,
    mode: str = "memo",
    source: str = "manual",
    context: dict | None = None,
) -> dict:
    mode_key = str(mode or "memo").strip().lower()
    return {
        "subject": spec.subject,
        "mode": mode_key,
        "source": source,
        "goals": spec.goals,
        "risk_tolerance": spec.risk_tolerance,
        "time_horizon": spec.time_horizon,
        "as_of_date": spec.as_of_date,
        "comparison_target": spec.comparison_target,
        "prompt": build_prompt_for_mode(mode_key, spec),
        "context": context or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate structured equity research prompts for the dashboard or external LLM workflows.")
    parser.add_argument("--ticker", required=True, help="Ticker or company name")
    parser.add_argument("--goals", default="long-term capital appreciation")
    parser.add_argument("--risk", default="moderate")
    parser.add_argument("--horizon", default="5+ years")
    parser.add_argument("--as-of-date", default="today")
    parser.add_argument("--compare", default="")
    parser.add_argument(
        "--mode",
        choices=["memo", "earnings", "valuation", "debate"],
        default="memo",
    )
    args = parser.parse_args()

    spec = ResearchPromptSpec(
        subject=args.ticker,
        goals=args.goals,
        risk_tolerance=args.risk,
        time_horizon=args.horizon,
        as_of_date=args.as_of_date,
        comparison_target=args.compare,
    )

    print(build_prompt_for_mode(args.mode, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
