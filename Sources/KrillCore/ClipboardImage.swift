import AppKit
import Foundation

/// Reads an image directly from the macOS pasteboard. Screenshot tools commonly
/// put a transient TIFF representation on the pasteboard, so TIFF is converted
/// to PNG before it reaches the model-facing media pipeline.
public enum ClipboardImage {
    /// Return a model-supported image container from the general pasteboard, if
    /// it currently holds one. Native PNG is preserved; TIFF is re-encoded as
    /// PNG because TIFF is not one of Krill's image attachment formats.
    public static func imageData(from pasteboard: NSPasteboard = .general) -> Data? {
        let png = pasteboard.data(forType: .png)
        let tiff = pasteboard.data(forType: .tiff)
            ?? NSImage(pasteboard: pasteboard)?.tiffRepresentation
        return normalizedImageData(pngData: png, tiffData: tiff)
    }

    /// Normalize pasteboard representations without accessing the system
    /// clipboard. This seam keeps conversion tests from mutating the user's
    /// pasteboard.
    static func normalizedImageData(pngData: Data?, tiffData: Data?) -> Data? {
        if let pngData, MediaAttachment.isImageData(pngData) { return pngData }
        guard let tiffData,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let png = bitmap.representation(using: .png, properties: [:]),
              MediaAttachment.isImageData(png) else { return nil }
        return png
    }
}
