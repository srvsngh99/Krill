import AppKit

/// The panel's animated presence: a soft gradient orb in the Krill wordmark
/// palette that reflects the conversation state. Purely presentational — it
/// receives state and microphone level from the controller and never touches
/// audio, recognition, or the agent loop.
///
/// States map to motion, not color changes (the brand gradient is constant):
///  - idle:      dim, slow breathing
///  - listening: brighter, breathing plus a scale response to the mic level
///  - thinking:  the highlight orbits the core while the agent works
///  - speaking:  a steady metronome pulse while narration plays
@MainActor
final class VoiceOrbView: NSView {
    enum State {
        case idle
        case listening
        case thinking
        case speaking
    }

    private let glowLayer = CALayer()
    private let coreLayer = CAGradientLayer()
    private let highlightLayer = CAGradientLayer()
    private var timer: Timer?
    private var phase: Double = 0
    private var orbitPhase: Double = 0

    private var state: State = .idle
    /// Smoothed 0…1 microphone energy (attack fast, release slow, like a meter).
    private var level: Double = 0
    private var displayedScale: Double = 1

    // Krill wordmark gradient: warm yellow -> ember -> raspberry.
    private static let brandColors: [CGColor] = [
        NSColor(srgbRed: 1.00, green: 0.78, blue: 0.36, alpha: 1).cgColor,
        NSColor(srgbRed: 1.00, green: 0.49, blue: 0.36, alpha: 1).cgColor,
        NSColor(srgbRed: 0.84, green: 0.25, blue: 0.43, alpha: 1).cgColor,
    ]
    private static let ember = NSColor(srgbRed: 1.00, green: 0.49, blue: 0.36, alpha: 1)

    private let orbDiameter: CGFloat = 84

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true

        glowLayer.backgroundColor = Self.ember.withAlphaComponent(0.9).cgColor
        glowLayer.shadowColor = Self.ember.cgColor
        glowLayer.shadowOpacity = 0.55
        glowLayer.shadowRadius = 22
        glowLayer.shadowOffset = .zero

        coreLayer.type = .radial
        coreLayer.colors = Self.brandColors
        coreLayer.locations = [0.0, 0.55, 1.0]
        coreLayer.startPoint = CGPoint(x: 0.35, y: 0.65)
        coreLayer.endPoint = CGPoint(x: 1.0, y: 0.0)

        // A soft off-center white highlight; orbiting it makes "thinking"
        // read as motion without changing the brand colors.
        highlightLayer.type = .radial
        highlightLayer.colors = [
            NSColor.white.withAlphaComponent(0.55).cgColor,
            NSColor.white.withAlphaComponent(0.0).cgColor,
        ]
        highlightLayer.startPoint = CGPoint(x: 0.3, y: 0.7)
        highlightLayer.endPoint = CGPoint(x: 0.9, y: 0.1)

        layer?.addSublayer(glowLayer)
        layer?.addSublayer(coreLayer)
        layer?.addSublayer(highlightLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("not used") }

    override var intrinsicContentSize: NSSize {
        NSSize(width: NSView.noIntrinsicMetric, height: orbDiameter + 28)
    }

    override func layout() {
        super.layout()
        let d = orbDiameter
        let rect = NSRect(x: (bounds.width - d) / 2, y: (bounds.height - d) / 2, width: d, height: d)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        for sub in [glowLayer, coreLayer, highlightLayer] {
            sub.frame = rect
            sub.cornerRadius = d / 2
        }
        CATransaction.commit()
    }

    func setState(_ newState: State) {
        guard state != newState else { return }
        state = newState
    }

    /// Feed the endpoint detector's decibel reading (about -60 dBFS silence to
    /// 0 dBFS clipping). Attack is immediate; release decays in `tick()`.
    func setLevel(decibels: Float) {
        let normalized = max(0, min(1, (Double(decibels) + 60) / 60))
        level = max(level, normalized)
    }

    func start() {
        guard timer == nil else { return }
        // 30 fps is plenty for a soft blob and keeps the panel's energy cost
        // negligible next to capture + recognition.
        let t = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        phase += 1.0 / 30.0
        level *= 0.88   // meter release

        let breathing = sin(phase * 2 * .pi / 3.4) * 0.02   // ~3.4 s cycle
        let targetScale: Double
        let targetGlow: Float
        switch state {
        case .idle:
            targetScale = 0.94 + breathing
            targetGlow = 0.25
        case .listening:
            targetScale = 1.0 + breathing + level * 0.16
            targetGlow = 0.55 + Float(level) * 0.35
        case .thinking:
            orbitPhase += 1.0 / 30.0
            targetScale = 0.98 + breathing
            targetGlow = 0.45
        case .speaking:
            targetScale = 1.0 + sin(phase * 2 * .pi / 0.9) * 0.05   // metronome
            targetGlow = 0.6
        }
        displayedScale += (targetScale - displayedScale) * 0.25   // ease toward target

        // Orbit the highlight while thinking; park it otherwise.
        let angle = orbitPhase * 2 * .pi / 2.6
        let hx = 0.5 + cos(angle) * 0.25
        let hy = 0.5 + sin(angle) * 0.25

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        let transform = CATransform3DMakeScale(displayedScale, displayedScale, 1)
        glowLayer.transform = transform
        coreLayer.transform = transform
        highlightLayer.transform = transform
        glowLayer.shadowOpacity = targetGlow
        if case .thinking = state {
            highlightLayer.startPoint = CGPoint(x: hx, y: hy)
        } else {
            highlightLayer.startPoint = CGPoint(x: 0.3, y: 0.7)
        }
        CATransaction.commit()
    }
}
