import Foundation
import NIOCore
import NIOHTTP1

/// Static shell for the phone/web agent UI, embedded in the binary so every
/// `krill serve` ships it with no packaging step (no resource bundle to lose
/// in `make dist`). The app page, the PWA manifest, and the brand icons.
/// The shell is served unauthenticated (it contains no data); the page asks
/// for the API key and sends it as a Bearer token on every API call.
///
/// Brand: the page and the icons follow the SAI suite spine + the Krill kit
/// (`sai-brand-kit/SUITE-DESIGN.md`, `sai-brand-kit/krill/DESIGN.md`) — ink on
/// paper chrome, dark #161310 terminal panels, Ember colour on the symbol
/// only, Space Grotesk + JetBrains Mono, hairline borders, corners <= 6px,
/// zero elevation. Icon payloads are the kit's own PNGs (`krill/app/*`).
///
/// The page source is maintained INLINE below as a Swift raw string. It is a
/// single self-contained file (no local assets beyond these routes), so
/// editing it is editing the string.
extension HTTPHandler {
    func handleWebUI(context: ChannelHandlerContext, method: HTTPMethod, path: String) {
        guard method == .GET || method == .HEAD else {
            return sendJSON(context: context, status: .methodNotAllowed,
                            body: ["error": "method not allowed"])
        }
        switch path {
        case "/ui", "/ui/":
            sendStatic(context: context, contentType: "text/html; charset=utf-8",
                       data: Data(WebUI.indexHTML.utf8))
        case "/ui/manifest.webmanifest":
            sendStatic(context: context, contentType: "application/manifest+json",
                       data: Data(WebUI.manifest.utf8))
        case "/ui/apple-touch-icon.png":
            sendStatic(context: context, contentType: "image/png", data: WebUI.appleTouchIcon,
                       cacheControl: "public, max-age=86400")
        case "/ui/icon-192.png", "/ui/icon.png":
            sendStatic(context: context, contentType: "image/png", data: WebUI.icon192,
                       cacheControl: "public, max-age=86400")
        case "/ui/icon-512.png":
            sendStatic(context: context, contentType: "image/png", data: WebUI.icon512,
                       cacheControl: "public, max-age=86400")
        case "/ui/maskable-512.png":
            sendStatic(context: context, contentType: "image/png", data: WebUI.maskable512,
                       cacheControl: "public, max-age=86400")
        default:
            sendJSON(context: context, status: .notFound, body: ["error": "Not found: \(path)"])
        }
    }

    private func sendStatic(
        context: ChannelHandlerContext, contentType: String, data: Data,
        cacheControl: String = "no-cache"
    ) {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: contentType)
        headers.add(name: "Content-Length", value: "\(data.count)")
        headers.add(name: "Cache-Control", value: cacheControl)
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        var buf = ByteBufferAllocator().buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }
}

enum WebUI {
    static let manifest = #"""
    {
      "name": "Krill",
      "short_name": "Krill",
      "start_url": "/ui",
      "scope": "/ui",
      "display": "standalone",
      "background_color": "#efece4",
      "theme_color": "#efece4",
      "icons": [
        { "src": "/ui/icon-192.png", "sizes": "192x192", "type": "image/png" },
        { "src": "/ui/icon-512.png", "sizes": "512x512", "type": "image/png" },
        { "src": "/ui/maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
      ]
    }
    """#

    // Brand icons, verbatim from sai-brand-kit/krill/app (the ink tile with
    // Ember code-lines). Regenerate by re-running the embed script against the
    // kit if the mark changes.
    static let appleTouchIcon = Data(base64Encoded: appleTouchB64) ?? Data()
    static let icon192 = Data(base64Encoded: icon192B64) ?? Data()
    static let icon512 = Data(base64Encoded: icon512B64) ?? Data()
    static let maskable512 = Data(base64Encoded: maskableB64) ?? Data()
    private static let appleTouchB64 = "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"
    private static let icon192B64 = "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"
    private static let icon512B64 = "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"
    private static let maskableB64 = "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"

    static let indexHTML = #"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover, interactive-widget=resizes-content">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Krill">
<meta name="theme-color" content="#efece4">
<link rel="manifest" href="/ui/manifest.webmanifest">
<link rel="apple-touch-icon" href="/ui/apple-touch-icon.png">
<link rel="icon" href="/ui/icon-192.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap">
<title>Krill</title>
<style>
/* SAI suite spine (sai-brand-kit/SUITE-DESIGN.md) + Krill kit (krill/DESIGN.md).
   Law: colour only in the symbol (Ember, on the tile + the mark's cursor);
   chrome/type ink on paper; dark #161310 terminal panels; hairline 1.5px
   borders; corners <=6px; zero elevation; whisper-dot texture. */
:root {
  --paper: #efece4; --ink: #1a1a1a; --dark: #161310; --mut: #6a655c;
  --line: #1a1a1a; --line-soft: rgba(26,26,26,.22);
  --panel-text: #e6e1d5; --panel-mut: #a49d8e;
  --ok: #1e7a3a; --warn: #a86a00; --err: #b3261e; --idle: #8b857a;
  --amber: #f5a623; --coral: #f4663c; --magenta: #e0457e;
  --sans: "Space Grotesk", -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --sat: env(safe-area-inset-top, 0px); --sab: env(safe-area-inset-bottom, 0px);
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html, body { height: 100%; }
body {
  margin: 0; background: var(--paper) radial-gradient(rgba(0,0,0,.045) 1px, transparent 1px);
  background-size: 5px 5px; color: var(--ink);
  font: 15.5px/1.5 var(--sans); letter-spacing: -0.01em;
  overscroll-behavior: none; -webkit-font-smoothing: antialiased; text-size-adjust: 100%;
}
button { font: inherit; color: inherit; background: none; border: 0; cursor: pointer; padding: 0; letter-spacing: inherit; }
input, textarea, select { font: inherit; color: var(--ink); letter-spacing: inherit; }
::selection { background: rgba(244,102,60,.3); }

/* // labels — mono, uppercase, letter-spaced */
.lbl, .microlbl { font-family: var(--mono); text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }

#app { height: 100dvh; display: flex; flex-direction: column; }
.view { flex: 1; min-height: 0; display: none; flex-direction: column; }
.view.active { display: flex; }

/* ---------- the mark ---------- */
.mark { font-family: var(--mono); font-weight: 700; font-size: 18px; letter-spacing: 0; }
.mark .cursor { color: var(--coral); animation: caret 1.1s steps(1) infinite; }
@keyframes caret { 50% { opacity: 0; } }

/* ---------- header ---------- */
.hdr {
  display: flex; align-items: center; gap: 10px;
  padding: calc(10px + var(--sat)) 16px 10px;
  background: var(--paper); border-bottom: 1.5px solid var(--line);
  flex: none; z-index: 5;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--idle); flex: none; }
.dot.ready { background: var(--ok); }
.dot.idle { background: var(--warn); }
.dot.down { background: var(--err); }
.hdr .chip {
  font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: .1em;
  color: var(--mut); border: 1.5px solid var(--line-soft); padding: 3px 8px; border-radius: 4px;
  max-width: 42vw; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.spacer { flex: 1; }
.iconbtn {
  min-width: 40px; height: 40px; border-radius: 4px; display: grid; place-items: center;
  color: var(--ink); font-size: 18px; flex: none; font-family: var(--mono);
}
.iconbtn:active { background: rgba(26,26,26,.08); }

/* ---------- sessions: the rail ---------- */
.scroll { flex: 1; min-height: 0; overflow-y: auto; -webkit-overflow-scrolling: touch; }
.slist { padding: 0 0 calc(96px + var(--sab)); }
.raillbl { padding: 16px 16px 8px; font-size: 10.5px; color: var(--mut); }
.srow {
  display: block; width: 100%; text-align: left; padding: 13px 16px;
  border-bottom: 1.5px solid var(--line-soft); background: var(--paper);
}
.srow:active { background: rgba(26,26,26,.06); }
.srow .t { font-weight: 500; font-size: 15.5px; margin-bottom: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.srow .m { font-family: var(--mono); font-size: 10.5px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--mut); display: flex; gap: 8px; align-items: center; }
.st { display: inline-flex; align-items: center; gap: 5px; font-weight: 700; }
.st i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.st.idle { color: var(--idle); } .st.idle i { background: var(--idle); }
.st.running { color: var(--ok); } .st.running i { background: var(--ok); }
.st.waiting { color: var(--warn); } .st.waiting i { background: var(--warn); }
.st.cancelled { color: var(--err); } .st.cancelled i { background: var(--err); }
.empty { text-align: center; color: var(--mut); padding: 70px 30px; }
.empty .tile { width: 56px; height: 56px; margin: 0 auto 16px; display: block; }
.newbtn {
  position: absolute; right: 16px; bottom: calc(20px + var(--sab));
  background: var(--ink); color: var(--paper); border-radius: 4px;
  font-family: var(--mono); font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; padding: 13px 18px; z-index: 6;
}
.newbtn:active { opacity: .85; }

/* ---------- session view ---------- */
.transcript { padding: 16px 14px 10px; display: flex; flex-direction: column; gap: 12px; }
.msg.user {
  align-self: flex-end; max-width: 88%; background: var(--ink); color: var(--paper);
  border-radius: 4px; padding: 9px 13px; white-space: pre-wrap; overflow-wrap: break-word;
}
.msg.assistant { align-self: stretch; overflow-wrap: break-word; }
.msg.assistant .md > :first-child { margin-top: 0; }
.msg.assistant .md > :last-child { margin-bottom: 0; }
.md p { margin: .55em 0; }
.md h1, .md h2, .md h3 { font-size: 1.05em; margin: .8em 0 .35em; }
.md ul, .md ol { margin: .4em 0; padding-left: 1.4em; }
.md code { font-family: var(--mono); font-size: .85em; background: rgba(26,26,26,.07); padding: 1px 5px; border-radius: 3px; }
.md pre { background: var(--dark); color: var(--panel-text); border-radius: 4px; padding: 11px 13px; overflow-x: auto; }
.md pre code { background: none; padding: 0; font-size: 12.5px; line-height: 1.55; }
.md a { color: var(--ink); text-decoration-thickness: 1.5px; }
.note { align-self: center; color: var(--mut); font-family: var(--mono); font-size: 11.5px; text-align: center; }

.tool { align-self: stretch; border: 1.5px solid var(--line); border-radius: 6px; background: var(--paper); overflow: hidden; }
.tool > .head { display: flex; align-items: center; gap: 9px; padding: 10px 13px; width: 100%; text-align: left; min-width: 0; min-height: 40px; }
.tool .tname { font-family: var(--mono); font-weight: 700; font-size: 12px; flex: none; }
.tool .targ { color: var(--mut); font-family: var(--mono); font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.tool .tdot { width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--idle); }
.tool .tdot.ok { background: var(--ok); }
.tool .tdot.err { background: var(--err); }
.tool .tdot.spin { background: var(--warn); animation: caret 1s steps(1) infinite; }
.tool .chev { color: var(--mut); flex: none; font-size: 10px; transition: transform .15s; font-family: var(--mono); }
.tool.open .chev { transform: rotate(90deg); }
.tool .body { display: none; border-top: 1.5px solid var(--line-soft); }
.tool.open .body { display: block; }
.tool .body pre {
  margin: 0 13px 11px; padding: 10px 12px; font-family: var(--mono); font-size: 11.5px;
  line-height: 1.55; overflow-x: auto; white-space: pre-wrap; overflow-wrap: break-word;
  max-height: 300px; overflow-y: auto; background: var(--dark); color: var(--panel-text); border-radius: 4px;
}
.tool .body pre.err { color: #ff9d94; }
.tool .body .lbl { font-size: 9.5px; color: var(--mut); padding: 10px 13px 6px; }

.approval { align-self: stretch; border: 2px solid var(--line); border-radius: 6px; padding: 12px 14px; background: var(--paper); }
.approval .lbl { font-size: 9.5px; color: var(--mut); margin-bottom: 6px; }
.approval .q { font-size: 14px; margin-bottom: 4px; }
.approval .q b { font-family: var(--mono); font-size: 12.5px; }
.approval pre {
  font-family: var(--mono); font-size: 11.5px; background: var(--dark); color: var(--panel-text);
  border-radius: 4px; padding: 9px 11px; margin: 8px 0 11px; overflow-x: auto;
  white-space: pre-wrap; overflow-wrap: break-word; max-height: 180px; overflow-y: auto;
}
.approval .row { display: flex; gap: 8px; }
.abtn {
  flex: 1; min-height: 42px; padding: 9px 4px; border-radius: 4px; font-family: var(--mono);
  font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
  border: 1.5px solid var(--line);
}
.abtn:active { opacity: .8; }
.abtn.deny { color: var(--err); border-color: var(--err); }
.abtn.once { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.approval.resolved { opacity: .5; }
.approval.resolved .row { display: none; }
.approval .verdict { font-family: var(--mono); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }

.working { display: none; align-items: center; gap: 8px; color: var(--mut); font-family: var(--mono);
  font-size: 11.5px; text-transform: uppercase; letter-spacing: .1em; padding: 2px 4px; align-self: flex-start; }
.working.on { display: flex; }
.working .blk { width: 8px; height: 14px; background: var(--ink); animation: caret 1s steps(1) infinite; }

/* ---------- composer ---------- */
.composer {
  flex: none; padding: 10px 12px calc(10px + var(--sab));
  background: var(--paper); border-top: 1.5px solid var(--line);
  display: flex; gap: 9px; align-items: flex-end;
}
.composer textarea {
  flex: 1; background: var(--paper); border: 1.5px solid var(--line); border-radius: 4px;
  padding: 9px 12px; resize: none; max-height: 132px; min-height: 42px; outline: none;
  line-height: 1.4; font-size: 16px;
}
.composer textarea::placeholder { color: var(--mut); }
.sendbtn {
  width: 42px; height: 42px; border-radius: 4px; background: var(--ink); color: var(--paper);
  display: grid; place-items: center; font-size: 17px; flex: none; font-family: var(--mono);
}
.sendbtn:active { opacity: .85; }

/* ---------- sheets ---------- */
.sheetwrap { position: fixed; inset: 0; z-index: 40; display: none; }
.sheetwrap.open { display: block; }
.sheetwrap .dimmer { position: absolute; inset: 0; background: rgba(26,26,26,.45); }
.sheet {
  position: absolute; left: 0; right: 0; bottom: 0; max-height: 88dvh;
  background: var(--paper); border-top: 1.5px solid var(--line);
  border-radius: 6px 6px 0 0;
  padding: 14px 18px calc(20px + var(--sab)); display: flex; flex-direction: column;
  overflow-y: auto;
}
.sheet > * { flex: none; }
.sheet h2 { font-size: 17px; font-weight: 700; margin: 0 0 14px; }
.sheet .lbl { font-size: 10px; color: var(--mut); margin: 14px 0 7px; }
.seg { display: flex; border: 1.5px solid var(--line); border-radius: 4px; overflow: hidden; }
.seg button {
  flex: 1; min-height: 40px; padding: 8px 2px; font-family: var(--mono); font-size: 10.5px;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 700; color: var(--mut);
  border-right: 1.5px solid var(--line-soft);
}
.seg button:last-child { border-right: 0; }
.seg button.on { background: var(--ink); color: var(--paper); }
.select, .field {
  width: 100%; background: var(--paper); border: 1.5px solid var(--line); border-radius: 4px;
  padding: 10px 12px; outline: none; appearance: none; -webkit-appearance: none; font-size: 15px;
}
.primary {
  width: 100%; background: var(--ink); color: var(--paper); font-family: var(--mono);
  font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: .1em;
  min-height: 48px; padding: 13px; border-radius: 4px; margin-top: 20px;
}
.primary:active { opacity: .85; }

/* workspace browser */
.wsbar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.wspath { font-family: var(--mono); font-size: 11px; color: var(--mut); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left; flex: 1; }
.wslist { overflow-y: auto; max-height: 32dvh; border: 1.5px solid var(--line); border-radius: 4px; }
.wsrow { display: flex; align-items: center; gap: 10px; width: 100%; text-align: left;
  padding: 11px 13px; min-height: 42px; border-bottom: 1.5px solid var(--line-soft); font-size: 14.5px; }
.wsrow:last-child { border-bottom: 0; }
.wsrow:active { background: rgba(26,26,26,.06); }
.wsrow .glyph { font-family: var(--mono); color: var(--mut); font-size: 12px; flex: none; }
.wsrow .repo { margin-left: auto; font-family: var(--mono); font-size: 9.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em; border: 1.5px solid var(--line);
  padding: 1px 6px; border-radius: 3px; flex: none; }
.wsempty { padding: 18px; color: var(--mut); font-family: var(--mono); font-size: 12px; text-align: center; }

/* ---------- connect ---------- */
.connect { flex: 1; display: flex; flex-direction: column; justify-content: center;
  padding: 34px 28px calc(34px + var(--sab)); max-width: 430px; margin: 0 auto; width: 100%; }
.connect .tile { width: 72px; height: 72px; margin: 0 auto 18px; display: block; }
.connect .bigmark { text-align: center; font-size: 30px; margin-bottom: 8px; }
.connect .sub { text-align: center; color: var(--ink); font-size: 15px; margin-bottom: 8px; }
.slab { background: var(--ink); color: var(--paper); padding: 0 .12em; }
.connect .byline { text-align: center; font-family: var(--mono); font-size: 9.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .28em; color: var(--mut); margin-bottom: 30px; }
.connect .lbl { font-size: 10px; color: var(--mut); margin: 14px 0 7px; }
.connect .field { font-family: var(--mono); font-size: 14px; }
.cerr { color: var(--err); font-family: var(--mono); font-size: 12px; margin-top: 14px; text-align: center; min-height: 20px; }

/* toast — ink on paper inverted */
#toast {
  position: fixed; left: 50%; transform: translateX(-50%); bottom: calc(90px + var(--sab));
  background: var(--ink); color: var(--paper); font-family: var(--mono);
  padding: 10px 16px; border-radius: 4px; font-size: 12px; z-index: 80;
  opacity: 0; transition: opacity .2s; pointer-events: none; max-width: 86vw;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#toast.show { opacity: 1; }

/* ---------- desktop: two-pane ---------- */
#pane-list { position: relative; }
.nosession { display: none; flex: 1; place-items: center; color: var(--mut); font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: .1em; }
@media (min-width: 920px) {
  #app.connected { flex-direction: row; }
  #app.connected #pane-list.active { width: 340px; flex: none; border-right: 1.5px solid var(--line); }
  #app.connected #nosession { display: grid; }
  #pane-session.active ~ #nosession { display: none; }
  #pane-session .hdr .back { display: none; }
  .msg.assistant, .tool, .approval { max-width: 780px; }
  .msg.user { max-width: 680px; }
  .transcript { padding: 22px 26px; }
  .composer { padding: 12px 26px calc(12px + var(--sab)); }
  .sheet { left: 50%; right: auto; bottom: auto; top: 50%; transform: translate(-50%,-50%);
    width: 460px; border-radius: 6px; border: 1.5px solid var(--line); }
}
</style>
</head>
<body>
<div id="app">

  <!-- connect -->
  <div class="view" id="view-connect">
    <div class="connect">
      <svg class="tile" viewBox="0 0 64 64" aria-hidden="true">
        <rect width="64" height="64" rx="6" fill="#1a1a1a"/>
        <polyline points="16,16 31,32 16,48" fill="none" stroke="#efece4" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="37" y="21" width="15" height="5" rx="2.5" fill="#ffb84d"/>
        <rect x="37" y="29.5" width="10" height="5" rx="2.5" fill="#ff7d5c"/>
        <rect x="37" y="38" width="17" height="5" rx="2.5" fill="#ff6aa6"/>
      </svg>
      <div class="mark bigmark">&#8250;Krill<span class="cursor">_</span></div>
      <div class="sub">The local engine that runs models and <span class="slab">codes</span></div>
      <div class="byline">A Sourav AI Labs project</div>
      <div class="lbl">// server</div>
      <input class="field" id="c-server" autocapitalize="off" autocorrect="off" placeholder="http://your-mac:57455">
      <div class="lbl">// api key</div>
      <input class="field" id="c-key" type="password" autocapitalize="off" placeholder="KRILL_API_KEY (blank if none)">
      <button class="primary" id="c-go">Connect</button>
      <div class="cerr" id="c-err"></div>
    </div>
  </div>

  <!-- sessions -->
  <div class="view" id="pane-list">
    <div class="hdr">
      <div class="dot" id="orb"></div>
      <div class="mark">&#8250;Krill<span class="cursor">_</span></div>
      <span class="chip" id="model-chip">connecting&#8230;</span>
      <div class="spacer"></div>
      <button class="iconbtn" id="btn-settings" aria-label="Settings">&#8942;</button>
    </div>
    <div class="scroll" id="slist-scroll">
      <div class="raillbl lbl">// sessions</div>
      <div class="slist" id="slist"></div>
    </div>
    <button class="newbtn" id="btn-new">+ New session</button>
  </div>

  <!-- session -->
  <div class="view" id="pane-session">
    <div class="hdr">
      <button class="iconbtn back" id="btn-back" aria-label="Back">&#8249;</button>
      <div style="min-width:0;flex:1">
        <div id="s-title" style="font-weight:500;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">Session</div>
        <div id="s-sub" class="microlbl" style="font-size:9.5px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:400"></div>
      </div>
      <span class="st idle microlbl" id="s-pill" style="font-size:10px"><i></i><span>idle</span></span>
      <button class="iconbtn" id="btn-more" aria-label="More">&#8943;</button>
    </div>
    <div class="scroll" id="t-scroll">
      <div class="transcript" id="transcript"></div>
      <div class="working" id="working" style="margin:0 18px 14px"><span class="blk"></span><span id="working-label">working&#8230;</span></div>
    </div>
    <div class="composer">
      <textarea id="input" rows="1" placeholder="Message the agent&#8230;" enterkeyhint="send"></textarea>
      <button class="sendbtn" id="btn-send" aria-label="Send">&#8593;</button>
    </div>
  </div>
  <div class="nosession" id="nosession"><div>Select or create a session</div></div>

</div>

<!-- new session sheet -->
<div class="sheetwrap" id="sheet-new">
  <div class="dimmer" data-close></div>
  <div class="sheet">
    <h2>New session</h2>
    <div class="lbl">// workspace</div>
    <div class="wsbar">
      <button class="iconbtn" id="ws-up" style="min-width:32px;height:32px" aria-label="Up">&#8249;</button>
      <div class="wspath" id="ws-path">~</div>
    </div>
    <div class="wslist" id="ws-list"></div>
    <div class="lbl">// model</div>
    <select class="select" id="new-model"></select>
    <div class="lbl">// permissions</div>
    <div class="seg" id="new-mode">
      <button data-v="plan">Plan</button>
      <button data-v="ask" class="on">Ask</button>
      <button data-v="accept-edits">Edits</button>
      <button data-v="accept-all">Auto</button>
    </div>
    <button class="primary" id="new-go">Start session</button>
  </div>
</div>

<!-- session actions sheet -->
<div class="sheetwrap" id="sheet-more">
  <div class="dimmer" data-close></div>
  <div class="sheet">
    <h2 id="more-title">Session</h2>
    <button class="abtn" id="more-cancel" style="margin-bottom:10px;padding:13px">Stop current run</button>
    <button class="abtn deny" id="more-delete" style="padding:13px">Delete session</button>
  </div>
</div>

<div id="toast"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const store = {
  load() { try { return JSON.parse(localStorage.getItem("krill.ui")) || {}; } catch { return {}; } },
  save(v) { localStorage.setItem("krill.ui", JSON.stringify(v)); },
};
let cfg = store.load();          // {server, key}
let sessions = [];
let cur = null;                  // {id, meta, lastSeq, tools:Map, aborter}
let listTimer = null;

/* ---------------- transport ---------------- */
function base() { return (cfg.server || location.origin).replace(/\/+$/, ""); }
function hdrs(json) {
  const h = {};
  if (cfg.key) h["Authorization"] = "Bearer " + cfg.key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
async function api(method, path, body) {
  const r = await fetch(base() + path, {
    method, headers: hdrs(body !== undefined),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await r.json(); } catch {}
  if (!r.ok) throw new Error((data && data.error) || (method + " " + path + " -> " + r.status));
  return data;
}

/* ---------------- tiny markdown ---------------- */
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, (_, c) => "<code>" + c + "</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|\W)\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src) {
  const lines = esc(src).split("\n");
  let out = "", i = 0, para = [];
  const flush = () => { if (para.length) { out += "<p>" + inlineMd(para.join("<br>")) + "</p>"; para = []; } };
  while (i < lines.length) {
    const l = lines[i];
    const fence = l.match(/^```(\w*)\s*$/);
    if (fence) {
      flush();
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out += "<pre><code>" + buf.join("\n") + "</code></pre>";
      continue;
    }
    const h = l.match(/^(#{1,3})\s+(.*)$/);
    if (h) { flush(); out += "<h" + h[1].length + ">" + inlineMd(h[2]) + "</h" + h[1].length + ">"; i++; continue; }
    const li = l.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      flush();
      let items = "";
      while (i < lines.length) {
        const m = lines[i].match(/^\s*[-*]\s+(.*)$/);
        if (!m) break;
        items += "<li>" + inlineMd(m[1]) + "</li>"; i++;
      }
      out += "<ul>" + items + "</ul>";
      continue;
    }
    const ol = l.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ol) {
      flush();
      let items = "";
      while (i < lines.length) {
        const m = lines[i].match(/^\s*\d+[.)]\s+(.*)$/);
        if (!m) break;
        items += "<li>" + inlineMd(m[1]) + "</li>"; i++;
      }
      out += "<ol>" + items + "</ol>";
      continue;
    }
    if (l.trim() === "") { flush(); } else { para.push(l); }
    i++;
  }
  flush();
  return out;
}

/* ---------------- view switching ---------------- */
const isDesktop = () => matchMedia("(min-width: 920px)").matches;
function show(view) {
  $("app").classList.toggle("connected", view !== "connect");
  for (const v of ["view-connect", "pane-list", "pane-session"]) $(v).classList.remove("active");
  if (view === "connect") { $("view-connect").classList.add("active"); return; }
  if (isDesktop()) {
    $("pane-list").classList.add("active");
    if (view === "session" && cur) $("pane-session").classList.add("active");
  } else {
    $(view === "session" ? "pane-session" : "pane-list").classList.add("active");
  }
}

/* ---------------- toast ---------------- */
let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ---------------- status + sessions ---------------- */
async function refreshStatus() {
  try {
    const s = await api("GET", "/v1/status");
    $("orb").className = "dot " + (s.model_loaded ? "ready" : "idle");
    $("model-chip").textContent = s.model || "no model loaded";
    return s;
  } catch (e) {
    $("orb").className = "dot down";
    $("model-chip").textContent = "unreachable";
    return null;
  }
}
function rel(iso) {
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 60) return "now";
  if (d < 3600) return Math.floor(d / 60) + "m";
  if (d < 86400) return Math.floor(d / 3600) + "h";
  return Math.floor(d / 86400) + "d";
}
function baseName(p) { const parts = p.split("/").filter(Boolean); return parts[parts.length - 1] || "/"; }
function stHTML(status) {
  return '<span class="st ' + status + '"><i></i><span>' + status + "</span></span>";
}
async function refreshSessions() {
  try {
    const r = await api("GET", "/v1/agent/sessions");
    sessions = r.sessions || [];
  } catch (e) { return; }
  const el = $("slist");
  if (!sessions.length) {
    el.innerHTML = '<div class="empty">' +
      '<svg class="tile" viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="6" fill="#1a1a1a"/><polyline points="16,16 31,32 16,48" fill="none" stroke="#efece4" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/><rect x="37" y="21" width="15" height="5" rx="2.5" fill="#ffb84d"/><rect x="37" y="29.5" width="10" height="5" rx="2.5" fill="#ff7d5c"/><rect x="37" y="38" width="17" height="5" rx="2.5" fill="#ff6aa6"/></svg>' +
      "<div style='font-weight:700'>No sessions yet</div><div style='font-size:13px;margin-top:6px'>" +
      "Point the agent at a folder and give it a task.</div></div>";
    return;
  }
  el.innerHTML = "";
  for (const s of sessions) {
    const b = document.createElement("button");
    b.className = "srow";
    b.innerHTML =
      '<div class="t">' + esc(s.title || "New session") + "</div>" +
      '<div class="m">' + stHTML(s.status) +
      "<span>" + esc(baseName(s.workspace)) + "</span><span>&#183;</span><span>" + rel(s.created_at) + "</span></div>";
    b.onclick = () => openSession(s.id);
    el.appendChild(b);
  }
}

/* ---------------- transcript rendering ---------------- */
function tScroll() { return $("t-scroll"); }
function nearBottom() { const s = tScroll(); return s.scrollHeight - s.scrollTop - s.clientHeight < 140; }
function toBottom() { const s = tScroll(); s.scrollTop = s.scrollHeight; }

function argSummary(name, argsJSON) {
  try {
    const a = JSON.parse(argsJSON);
    return a.command || a.file_path || a.path || a.pattern || a.url || a.query
      || Object.values(a).find((v) => typeof v === "string") || "";
  } catch { return argsJSON; }
}
function addUser(text) {
  const d = document.createElement("div");
  d.className = "msg user"; d.textContent = text;
  $("transcript").appendChild(d);
}
function addAssistant(text) {
  const d = document.createElement("div");
  d.className = "msg assistant";
  d.innerHTML = '<div class="md">' + md(text) + "</div>";
  $("transcript").appendChild(d);
}
function addNote(text) {
  const d = document.createElement("div");
  d.className = "note"; d.textContent = text;
  $("transcript").appendChild(d);
}
function addToolCall(name, args) {
  const d = document.createElement("div");
  d.className = "tool";
  const pretty = (() => { try { return JSON.stringify(JSON.parse(args), null, 2); } catch { return args; } })();
  d.innerHTML =
    '<button class="head"><span class="tdot spin"></span><span class="tname">' + esc(name) + "</span>" +
    '<span class="targ">' + esc(argSummary(name, args)) + "</span>" +
    '<span class="chev">&#9654;</span></button>' +
    '<div class="body"><div class="lbl">// arguments</div><pre>' + esc(pretty) + "</pre>" +
    '<div class="result"></div></div>';
  d.querySelector(".head").onclick = () => d.classList.toggle("open");
  $("transcript").appendChild(d);
  cur.tools.set(name, d);   // latest open card per tool name
  return d;
}
function addToolResult(name, content, isError) {
  let card = cur.tools.get(name);
  if (!card || card.dataset.done) card = addToolCall(name, "{}");
  card.dataset.done = "1";
  cur.tools.delete(name);
  card.querySelector(".tdot").className = "tdot " + (isError ? "err" : "ok");
  const clip = content.length > 4000 ? content.slice(0, 4000) + "\n… (" + content.length + " chars)" : content;
  card.querySelector(".result").innerHTML =
    '<div class="lbl">// result</div><pre class="' + (isError ? "err" : "") + '">' + esc(clip) + "</pre>";
  if (isError) card.classList.add("open");
}
function addApproval(ev) {
  const d = document.createElement("div");
  d.className = "approval"; d.dataset.aid = ev.id;
  const pretty = (() => { try { return JSON.stringify(JSON.parse(ev.args), null, 2); } catch { return ev.args; } })();
  d.innerHTML =
    '<div class="lbl">// approval</div>' +
    '<div class="q">Run <b>' + esc(ev.tool) + "</b>?</div><pre>" + esc(pretty) + "</pre>" +
    '<div class="row"><button class="abtn deny">Deny</button>' +
    '<button class="abtn once">Allow</button>' +
    '<button class="abtn always">Always</button></div>' +
    '<div class="verdict"></div>';
  const answer = async (allow, always) => {
    try { await api("POST", "/v1/agent/sessions/" + cur.id + "/approvals", { id: ev.id, allow, always }); }
    catch (e) { toast(e.message); }
  };
  d.querySelector(".deny").onclick = () => answer(false, false);
  d.querySelector(".once").onclick = () => answer(true, false);
  d.querySelector(".always").onclick = () => answer(true, true);
  $("transcript").appendChild(d);
}
function resolveApproval(ev) {
  const d = document.querySelector('.approval[data-aid="' + CSS.escape(ev.id) + '"]');
  if (!d) return;
  d.classList.add("resolved");
  d.querySelector(".verdict").textContent = ev.allow ? "allowed" : "denied";
  d.querySelector(".verdict").style.color = ev.allow ? "var(--ok)" : "var(--err)";
}
function setSessionStatus(status) {
  if (!cur) return;
  cur.meta.status = status;
  $("s-pill").className = "st " + status + " microlbl";
  $("s-pill").querySelector("span").textContent = status;
  const running = status === "running" || status === "waiting";
  $("working").classList.toggle("on", running);
  $("working-label").textContent = status === "waiting" ? "waiting for approval…" : "working…";
  $("btn-send").innerHTML = running ? "&#9632;" : "&#8593;";
}
function applyEvent(ev) {
  switch (ev.type) {
    case "user": addUser(ev.text); break;
    case "assistant": case "assistant_final": addAssistant(ev.text); break;
    case "tool_call": addToolCall(ev.name, ev.args); break;
    case "tool_result": addToolResult(ev.name, ev.content, ev.is_error); break;
    case "note": addNote(ev.text); break;
    case "status": setSessionStatus(ev.status); break;
    case "approval_request": addApproval(ev); break;
    case "approval_resolved": resolveApproval(ev); break;
  }
}

/* ---------------- SSE tail ---------------- */
async function tail(sessionId) {
  while (cur && cur.id === sessionId) {
    const aborter = new AbortController();
    cur.aborter = aborter;
    try {
      const r = await fetch(base() + "/v1/agent/sessions/" + sessionId + "/events?since=" + cur.lastSeq,
        { headers: hdrs(false), signal: aborter.signal });
      if (!r.ok) throw new Error("events " + r.status);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
          let seq = null, data = null;
          for (const line of chunk.split("\n")) {
            if (line.startsWith("id: ")) seq = parseInt(line.slice(4), 10);
            else if (line.startsWith("data: ")) data = line.slice(6);
          }
          if (data == null) continue;   // heartbeat comment
          if (!cur || cur.id !== sessionId) return;
          if (seq != null) {
            if (seq <= cur.lastSeq) continue;   // duplicate on reconnect
            cur.lastSeq = seq;
          }
          try {
            const keep = nearBottom();
            applyEvent(JSON.parse(data));
            if (keep) toBottom();
          } catch {}
        }
      }
    } catch (e) {
      if (aborter.signal.aborted) return;
    }
    if (!cur || cur.id !== sessionId) return;
    await new Promise((res) => setTimeout(res, 1500));   // reconnect backoff
  }
}

/* ---------------- session open/close ---------------- */
async function openSession(id) {
  closeSession();
  let detail;
  try { detail = await api("GET", "/v1/agent/sessions/" + id); }
  catch (e) { toast(e.message); return; }
  cur = { id, meta: detail, lastSeq: 0, tools: new Map(), aborter: null };
  $("transcript").innerHTML = "";
  $("s-title").textContent = detail.title || "New session";
  $("s-sub").textContent = baseName(detail.workspace) + " · " + detail.permission_mode
    + (detail.model ? " · " + detail.model : "");
  for (const ev of detail.events || []) {
    if (typeof ev.seq === "number") cur.lastSeq = Math.max(cur.lastSeq, ev.seq);
    applyEvent(ev);
  }
  setSessionStatus(detail.status);
  show("session");
  toBottom();
  history.pushState({ s: id }, "", "#/s/" + id);
  tail(id);
}
function closeSession() {
  if (cur && cur.aborter) cur.aborter.abort();
  cur = null;
}
function backToList() {
  closeSession();
  show("list");
  refreshSessions();
}

/* ---------------- composer ---------------- */
const input = $("input");
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 132) + "px";
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && isDesktop()) { e.preventDefault(); $("btn-send").click(); }
});
$("btn-send").onclick = async () => {
  if (!cur) return;
  const running = cur.meta.status === "running" || cur.meta.status === "waiting";
  if (running) {
    try { await api("POST", "/v1/agent/sessions/" + cur.id + "/cancel"); } catch (e) { toast(e.message); }
    return;
  }
  const text = input.value.trim();
  if (!text) return;
  input.value = ""; input.style.height = "auto";
  try { await api("POST", "/v1/agent/sessions/" + cur.id + "/messages", { text }); }
  catch (e) { toast(e.message); }
  toBottom();
};

/* ---------------- new session sheet ---------------- */
let ws = { path: "~", chosen: null };
async function loadWs(path) {
  let r;
  try { r = await api("GET", "/v1/agent/workspaces?path=" + encodeURIComponent(path)); }
  catch (e) { toast(e.message); return; }
  ws.path = r.path; ws.parent = r.parent; ws.chosen = r.path;
  $("ws-path").textContent = r.path;
  const list = $("ws-list");
  list.innerHTML = "";
  if (!r.dirs.length) list.innerHTML = '<div class="wsempty">No folders here — using this directory.</div>';
  for (const d of r.dirs) {
    const b = document.createElement("button");
    b.className = "wsrow";
    b.innerHTML = '<span class="glyph">&#8250;</span><span>' + esc(d.name) + "</span>" +
      (d.is_repo ? '<span class="repo">git</span>' : "");
    b.onclick = () => loadWs(d.path);
    list.appendChild(b);
  }
}
$("ws-up").onclick = () => { if (ws.parent) loadWs(ws.parent); };
$("new-mode").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  for (const x of $("new-mode").children) x.classList.remove("on");
  b.classList.add("on");
});
async function openNewSheet() {
  $("sheet-new").classList.add("open");
  loadWs(ws.path || "~");
  const sel = $("new-model");
  sel.innerHTML = '<option value="">Active model (server default)</option>';
  try {
    const s = await api("GET", "/v1/status");
    for (const m of s.installed_models || []) {
      const o = document.createElement("option");
      o.value = m; o.textContent = m + (s.model === m ? "  (loaded)" : "");
      if (s.model === m) o.selected = true;
      sel.appendChild(o);
    }
  } catch {}
}
$("new-go").onclick = async () => {
  const mode = $("new-mode").querySelector(".on").dataset.v;
  const body = { workspace: ws.chosen, permission_mode: mode };
  const model = $("new-model").value;
  if (model) body.model = model;
  let s;
  try { s = await api("POST", "/v1/agent/sessions", body); }
  catch (e) { toast(e.message); return; }
  $("sheet-new").classList.remove("open");
  await openSession(s.id);
  input.focus();
};

/* ---------------- more sheet ---------------- */
$("btn-more").onclick = () => { if (cur) { $("more-title").textContent = cur.meta.title || "Session"; $("sheet-more").classList.add("open"); } };
$("more-cancel").onclick = async () => {
  $("sheet-more").classList.remove("open");
  if (!cur) return;
  try { await api("POST", "/v1/agent/sessions/" + cur.id + "/cancel"); toast("stopping…"); }
  catch (e) { toast(e.message); }
};
$("more-delete").onclick = async () => {
  $("sheet-more").classList.remove("open");
  if (!cur) return;
  if (!confirm("Delete this session?")) return;
  try { await api("DELETE", "/v1/agent/sessions/" + cur.id); } catch (e) { toast(e.message); }
  backToList();
};

/* ---------------- wiring ---------------- */
for (const w of document.querySelectorAll(".sheetwrap")) {
  w.addEventListener("click", (e) => { if (e.target.dataset.close !== undefined) w.classList.remove("open"); });
}
$("btn-new").onclick = openNewSheet;
$("btn-back").onclick = () => { history.back(); };
$("btn-settings").onclick = () => { show("connect"); prefillConnect(); };
window.addEventListener("popstate", () => { if (cur) backToList(); });

function prefillConnect() {
  $("c-server").value = cfg.server || location.origin;
  $("c-key").value = cfg.key || "";
  $("c-err").textContent = "";
}
$("c-go").onclick = async () => {
  const server = $("c-server").value.trim().replace(/\/+$/, "") || location.origin;
  const key = $("c-key").value.trim();
  const probe = { server, key };
  const old = cfg; cfg = probe;
  try {
    await api("GET", "/v1/status");
  } catch (e) {
    cfg = old;
    $("c-err").textContent = "Could not connect: " + e.message;
    return;
  }
  store.save(cfg);
  boot();
};

async function boot() {
  prefillConnect();
  const s = await refreshStatus();
  if (s === null && !cfg.server && !cfg.key) { show("connect"); return; }
  if (s === null) { show("connect"); $("c-err").textContent = "Saved server is unreachable."; return; }
  show("list");
  refreshSessions();
  clearInterval(listTimer);
  listTimer = setInterval(() => {
    if (document.hidden) return;   // poll only while visible (fleet rule)
    refreshStatus();
    if (!cur || isDesktop()) refreshSessions();
  }, 8000);
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && cur) {
    // resume the tail fast after the phone unlocks
    if (cur.aborter) cur.aborter.abort();
    const id = cur.id;
    setTimeout(() => { if (cur && cur.id === id) tail(id); }, 50);
  }
});
boot();
</script>
</body>
</html>
"""#
}
