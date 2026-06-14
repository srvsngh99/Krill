#ifndef KRILLM_CEDITLINE_SHIM_H
#define KRILLM_CEDITLINE_SHIM_H
/* macOS ships libedit with a readline-compatible API. We use the readline
 * emulation (readline / add_history / completion hooks) for the interactive
 * REPL: line editing, history, and tab completion with no third-party dep. */
#include <editline/readline.h>

/* Thin setters for libedit's mutable completion globals. Swift 6 strict
 * concurrency rejects direct access to imported C global `var`s; routing the
 * writes through these inline C functions keeps the calls on the (single) REPL
 * thread without tripping that check. */
static inline void krillm_set_attempted_completion(CPPFunction *fn) {
    rl_attempted_completion_function = fn;
}
static inline void krillm_set_completion_over(int value) {
    rl_attempted_completion_over = value;
}
#endif
