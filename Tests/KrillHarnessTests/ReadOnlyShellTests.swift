import XCTest
@testable import KrillHarness

final class ReadOnlyShellTests: XCTestCase {
    func testDenied() {
        let denied = [
            "rm -rf x", "find . -delete", "find . -exec rm {} \\;", "find . -fprint f",
            "git diff > f", "git diff --output=f", "git diff --out=f", "git -c core.pager=rm log",
            "git grep -O x", "git grep -nOx y", "git branch -D main", "git branch newname",
            "git diff --ext-diff", "git commit -m x", "git",
            "ls; rm x", "ls && rm x", "ls || rm x", "ls & rm x", "ls |& cat",
            "echo `rm x`", "echo $(rm x)", "echo \"$(rm x)\"", "echo \"`rm x`\"", "echo ${x}", "echo $HOME",
            "sed -i s/a/b/ f", "sort -o f x", "sort -ro f x", "sort --output=f x", "sort --comp=rm x",
            "tree -o f", "tree -ao f", "rg --pre rm x", "rg --pre=rm x", "rg -z x", "rg --search-zip x",
            "cat x | sh", "cat x | xargs rm", "cat x | ", "| cat", "ls ||| cat",
            "X=1 ls", "ls *(e:'rm x':)", "ls *", "ls [a]", "cat <(rm x)", "ls\nrm x", "ls\rrm x",
            "ls \\\nrm x", "ls\\;rm x", "echo \"a\\b\"", "echo {a,b}", "(ls)",
            "uniq a b", "date -s 2020-01-01", "date 1112", "date --set=x", "echo hi > f", "cat < f",
            "tee f", "curl example.com", "python3 -c 'print(1)'", "/bin/ls", "./ls", "env ls", "time ls",
            "file -C", "", "   ", "echo 'unbalanced", "echo \"unbalanced",
        ]
        // Regression: a combining mark must not hide a quote or metacharacter from the scanner (sh sees bytes).
        let hidden = [
            "echo '" + "'\u{301}" + " ; touch PWNED ; echo '" + "'\u{301}",
            "echo hi >\u{301} PWNED2", "ls ;\u{301}", "ls ;\u{200D} rm x", "ls ;\u{FE0F} rm x",
            "ls '\u{200D}' ; rm x", "date -f -%Y -- -2020",
        ]
        for c in denied + hidden { XCTAssertNotNil(ReadOnlyShell.violation(c), "must deny: \(c)") }
    }

    func testAllowed() {
        let allowed = [
            "ls -la", "find . -name '*.swift' | wc -l", "grep -rn \"foo\" Sources",
            "rg -n 'func run' Sources", "git status", "git log --oneline -5",
            "git diff HEAD~1 --stat", "git branch -a", "git branch --list", "git ls-files -o",
            "cat README.md | head -20", "echo 'a;b'", "grep \"a|b\" f", "echo 'a$b `c` \\d'",
            "date +%Y", "sort -n f | head -3", "tree -L 2", "  ls   -l  ",
        ]
        for c in allowed { XCTAssertNil(ReadOnlyShell.violation(c), "must allow: \(c)") }
    }

    func testDenialNamesTheProblem() {
        XCTAssertEqual(ReadOnlyShell.violation("rm x"), "'rm' is not an allowed read-only command")
        XCTAssertTrue(ReadOnlyShell.violation("ls; ls")!.contains("';'"))
    }

    func testPolicyDecisions() {
        let plan = PermissionPolicy(mode: .plan)
        func d(_ p: PermissionPolicy, _ cmd: String?) -> PermissionDecision {
            p.decision(toolName: "bash", isReadOnly: false, bashCommand: cmd)
        }
        XCTAssertEqual(d(plan, "ls"), .allow)
        XCTAssertEqual(d(PermissionPolicy(mode: .adaptive), "ls"), .allow)
        guard case .deny(let reason) = d(plan, "rm x") else { return XCTFail("rm must be denied") }
        XCTAssertTrue(reason.contains("'rm' is not an allowed"))
        guard case .deny = d(plan, "") else { return XCTFail("empty must be denied") }
        guard case .deny = d(plan, nil) else { return XCTFail("no command must be denied") }
        XCTAssertEqual(d(PermissionPolicy(mode: .ask), "ls"), .ask)
        XCTAssertEqual(d(PermissionPolicy(mode: .acceptEdits), "ls"), .ask)
        guard case .deny = d(PermissionPolicy(mode: .plan, deny: ["bash"]), "ls") else {
            return XCTFail("deny list must win")
        }
    }
}
