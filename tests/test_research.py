from ai_trading.research import (
    ResearchPromptSpec,
    build_prompt_for_mode,
    build_research_packet,
    build_debate_prompt,
    build_earnings_prompt,
    build_investment_memo_prompt,
    build_valuation_prompt,
)


def test_build_investment_memo_prompt_includes_portfolio_fit_inputs() -> None:
    spec = ResearchPromptSpec(
        subject="NVDA",
        goals="compound capital with quality growth",
        risk_tolerance="moderate-high",
        time_horizon="7 years",
        as_of_date="June 11, 2026",
    )

    prompt = build_investment_memo_prompt(spec)

    assert "Produce an institutional-quality investment memo on NVDA." in prompt
    assert "Investment goals: compound capital with quality growth" in prompt
    assert "Risk tolerance: moderate-high" in prompt
    assert "Time horizon: 7 years" in prompt
    assert "Base your analysis on information available as of June 11, 2026." in prompt


def test_build_investment_memo_prompt_appends_comparison_when_present() -> None:
    spec = ResearchPromptSpec(
        subject="MSFT",
        comparison_target="GOOGL",
        as_of_date="June 11, 2026",
    )

    prompt = build_investment_memo_prompt(spec)

    assert "Also include a direct comparison between MSFT and GOOGL" in prompt
    assert "Conclude which is the better investment opportunity today and why." in prompt


def test_specialized_prompt_builders_reference_subject_and_date() -> None:
    spec = ResearchPromptSpec(subject="AMZN", as_of_date="June 11, 2026")

    earnings = build_earnings_prompt(spec)
    valuation = build_valuation_prompt(spec)
    debate = build_debate_prompt(spec)

    assert "Analyze the latest earnings report from AMZN." in earnings
    assert "Base your analysis on information available as of June 11, 2026." in earnings
    assert "Estimate the intrinsic value of AMZN as of June 11, 2026." in valuation
    assert "Act as two investors debating AMZN as of June 11, 2026." in debate


def test_build_prompt_for_mode_defaults_to_memo() -> None:
    spec = ResearchPromptSpec(subject="META", as_of_date="June 11, 2026")

    prompt = build_prompt_for_mode("unknown", spec)

    assert "Produce an institutional-quality investment memo on META." in prompt


def test_build_research_packet_includes_prompt_and_context() -> None:
    spec = ResearchPromptSpec(subject="TSLA", as_of_date="June 11, 2026")

    packet = build_research_packet(
        spec=spec,
        mode="debate",
        source="scanner_detail",
        context={"score": 78.5},
    )

    assert packet["subject"] == "TSLA"
    assert packet["mode"] == "debate"
    assert packet["source"] == "scanner_detail"
    assert packet["context"] == {"score": 78.5}
    assert "Act as two investors debating TSLA as of June 11, 2026." in packet["prompt"]
