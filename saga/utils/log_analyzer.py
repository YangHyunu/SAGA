"""SAGA Log Analyzer — 실시간 로그 분석 및 이슈 분류 에이전트.

Usage:
    python -m saga.utils.log_analyzer [logfile]
    # 또는 파이프:
    python -m saga | python -m saga.utils.log_analyzer --stdin
"""

import re
import sys
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# Pattern definitions
# ──────────────────────────────────────────────

PATTERNS = {
    # HTTP errors
    "http_401": re.compile(r'HTTP/1\.1 401 Unauthorized'),
    "http_404": re.compile(r'"[A-Z]+ .+ HTTP/1\.\d" 404'),
    "http_502": re.compile(r'"[A-Z]+ .+ HTTP/1\.\d" 502'),
    "http_429": re.compile(r'429 rate limit|HTTP/1\.\d" 429'),
    "http_500": re.compile(r'"[A-Z]+ .+ HTTP/1\.\d" 500'),

    # SAGA-specific
    "url_duplicate": re.compile(r'/v1/chat/completions.*?/v1/chat/completions'),
    "url_encoded_space": re.compile(r'%20'),
    "regex_parse_fail": re.compile(r'Regex parse failed'),
    "flash_extraction_fail": re.compile(r'Flash extraction (?:failed|also failed)'),
    "api_key_issue": re.compile(r'Unauthorized|api.key|API key', re.IGNORECASE),
    "curator_fail": re.compile(r'\[Curator\].*(?:failed|error|exception)', re.IGNORECASE),
    "curator_ok": re.compile(r'\[Curator\].*connected successfully'),
    "sub_b_error": re.compile(r'\[Sub-B\].*Error'),
    "timeout": re.compile(r'timeout|timed out', re.IGNORECASE),

    # Timing
    "timing": re.compile(r'\[TIMING\] (\w[\w ]*): (\d+)ms'),
    "total_timing": re.compile(r'\[TIMING\] Total: (\d+)ms'),

    # Session
    "session_created": re.compile(r'\[Session\] Created new session: (\w+)'),
    "turn_complete": re.compile(r'\[Sub-B\] Turn (\d+) post-processing complete'),
}

SEVERITY = {
    "http_401": "CRITICAL",
    "http_404": "ERROR",
    "http_502": "ERROR",
    "http_429": "WARN",
    "http_500": "CRITICAL",
    "url_duplicate": "ERROR",
    "url_encoded_space": "WARN",
    "regex_parse_fail": "WARN",
    "flash_extraction_fail": "WARN",
    "api_key_issue": "CRITICAL",
    "curator_fail": "ERROR",
    "curator_ok": "INFO",
    "sub_b_error": "ERROR",
    "timeout": "ERROR",
    "timing": "INFO",
    "total_timing": "INFO",
    "session_created": "INFO",
    "turn_complete": "INFO",
}

REMEDIATION = {
    "http_401": "API 키가 유효하지 않음. config.yaml의 api_keys 섹션 또는 환경변수 확인.",
    "http_404": "잘못된 URL 경로. 클라이언트(RisuAI 등)의 Base URL 설정 확인.",
    "http_502": "업스트림 LLM 호출 실패. 원인은 보통 401(인증) 또는 429(레이트리밋).",
    "http_429": "레이트 리밋 도달. 요청 간격을 늘리거나 API 플랜 업그레이드.",
    "http_500": "서버 내부 오류. 전체 traceback 확인 필요.",
    "url_duplicate": "클라이언트 Base URL에 path가 중복 포함됨. 'http://localhost:8000/v1'로만 설정.",
    "url_encoded_space": "URL에 공백 문자 포함. 클라이언트 설정에서 공백 제거.",
    "regex_parse_fail": "LLM 응답에 ```state 블록 없음. 시스템 프롬프트에 state block 지시 확인.",
    "flash_extraction_fail": "Flash(Gemini) 폴백 추출도 실패. extraction 모델 또는 API 키 확인.",
    "api_key_issue": "API 인증 실패. 환경변수 또는 config.yaml 확인: echo $ANTHROPIC_API_KEY",
    "curator_fail": "Letta 큐레이터 연결/실행 실패. docker-compose.letta.yaml 상태 확인.",
    "sub_b_error": "후처리(Sub-B) 에러. 로그의 traceback에서 상세 원인 확인.",
    "timeout": "요청 시간 초과. 네트워크 또는 LLM 서버 상태 확인.",
}


# ──────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────

class LogAnalyzer:
    def __init__(self):
        self.issues: list[dict] = []
        self.counters: Counter = Counter()
        self.timings: list[dict] = []
        self.sessions: set = set()
        self.turns_ok: int = 0
        self.lines_processed: int = 0

    def analyze_line(self, line: str) -> list[dict]:
        """Analyze a single log line. Returns list of detected issues."""
        self.lines_processed += 1
        detected = []

        for name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue

            self.counters[name] += 1

            if name == "timing":
                self.timings.append({
                    "component": match.group(1),
                    "ms": int(match.group(2)),
                })
                continue

            if name == "total_timing":
                self.timings.append({
                    "component": "Total",
                    "ms": int(match.group(1)),
                })
                continue

            if name == "session_created":
                self.sessions.add(match.group(1))
                continue

            if name == "turn_complete":
                self.turns_ok += 1
                continue

            if name == "curator_ok":
                continue

            issue = {
                "pattern": name,
                "severity": SEVERITY.get(name, "INFO"),
                "line": line.strip()[:200],
                "remediation": REMEDIATION.get(name, ""),
            }
            self.issues.append(issue)
            detected.append(issue)

        return detected

    def analyze_text(self, text: str) -> "LogAnalyzer":
        """Analyze multi-line log text."""
        for line in text.strip().split("\n"):
            self.analyze_line(line)
        return self

    def get_report(self) -> dict:
        """Generate analysis report."""
        # Deduplicate issues by pattern
        unique_issues = {}
        for issue in self.issues:
            key = issue["pattern"]
            if key not in unique_issues:
                unique_issues[key] = {
                    **issue,
                    "count": self.counters[key],
                    "first_occurrence": issue["line"],
                }
            else:
                unique_issues[key]["count"] = self.counters[key]

        # Sort by severity
        severity_order = {"CRITICAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3}
        sorted_issues = sorted(
            unique_issues.values(),
            key=lambda x: severity_order.get(x["severity"], 99),
        )

        # Timing stats
        timing_stats = {}
        for t in self.timings:
            comp = t["component"]
            if comp not in timing_stats:
                timing_stats[comp] = {"min": t["ms"], "max": t["ms"], "sum": t["ms"], "count": 1}
            else:
                s = timing_stats[comp]
                s["min"] = min(s["min"], t["ms"])
                s["max"] = max(s["max"], t["ms"])
                s["sum"] += t["ms"]
                s["count"] += 1
        for comp, s in timing_stats.items():
            s["avg"] = round(s["sum"] / s["count"])

        return {
            "summary": {
                "lines_processed": self.lines_processed,
                "sessions_seen": len(self.sessions),
                "turns_completed": self.turns_ok,
                "total_issues": len(self.issues),
                "critical": sum(1 for i in self.issues if i["severity"] == "CRITICAL"),
                "errors": sum(1 for i in self.issues if i["severity"] == "ERROR"),
                "warnings": sum(1 for i in self.issues if i["severity"] == "WARN"),
            },
            "issues": sorted_issues,
            "timing": timing_stats,
        }

    def print_report(self, file=sys.stdout):
        """Print human-readable report."""
        report = self.get_report()
        s = report["summary"]

        print("\n" + "=" * 60, file=file)
        print("  SAGA Log Analysis Report", file=file)
        print("=" * 60, file=file)

        print(f"\n  Lines: {s['lines_processed']} | Sessions: {s['sessions_seen']} | Turns OK: {s['turns_completed']}", file=file)
        print(f"  Issues: {s['total_issues']} (CRITICAL: {s['critical']}, ERROR: {s['errors']}, WARN: {s['warnings']})", file=file)

        if report["issues"]:
            print("\n" + "-" * 60, file=file)
            print("  Issues (by severity)", file=file)
            print("-" * 60, file=file)

            for issue in report["issues"]:
                sev = issue["severity"]
                marker = {"CRITICAL": "!!!", "ERROR": " ! ", "WARN": " ~ "}.get(sev, "   ")
                print(f"\n  [{marker}] {sev}: {issue['pattern']} (x{issue['count']})", file=file)
                print(f"        {issue['first_occurrence'][:120]}", file=file)
                if issue["remediation"]:
                    print(f"     -> {issue['remediation']}", file=file)

        if report["timing"]:
            print("\n" + "-" * 60, file=file)
            print("  Timing (ms)", file=file)
            print("-" * 60, file=file)
            print(f"  {'Component':<25} {'Avg':>6} {'Min':>6} {'Max':>6} {'Count':>6}", file=file)
            for comp, t in report["timing"].items():
                print(f"  {comp:<25} {t['avg']:>6} {t['min']:>6} {t['max']:>6} {t['count']:>6}", file=file)

        if not report["issues"]:
            print("\n  No issues detected.", file=file)

        print("\n" + "=" * 60 + "\n", file=file)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAGA Log Analyzer")
    parser.add_argument("logfile", nargs="?", help="Log file to analyze")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin (pipe mode)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Watch mode: tail -f style analysis")
    args = parser.parse_args()

    analyzer = LogAnalyzer()

    if args.stdin or (not args.logfile and not sys.stdin.isatty()):
        # Pipe mode
        for line in sys.stdin:
            detected = analyzer.analyze_line(line)
            if args.watch and detected:
                for d in detected:
                    sev = d["severity"]
                    print(f"[{sev}] {d['pattern']}: {d['line'][:100]}", file=sys.stderr)

    elif args.logfile:
        path = Path(args.logfile)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        analyzer.analyze_text(path.read_text())

    else:
        parser.print_help()
        sys.exit(1)

    if args.json:
        print(json.dumps(analyzer.get_report(), indent=2, ensure_ascii=False))
    else:
        analyzer.print_report()


if __name__ == "__main__":
    main()
