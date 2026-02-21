import 'dart:math' as math;
import 'dart:typed_data';

class EmbeddingUtils {
  /// Store embeddings as Float32List in SQLite BLOB.
  static Uint8List float32ToBytes(List<double> v) {
    final f32 = Float32List(v.length);
    for (var i = 0; i < v.length; i++) {
      f32[i] = v[i].toDouble();
    }
    return f32.buffer.asUint8List();
  }

  static List<double> l2Normalize(List<double> v) {
    double sumSq = 0.0;
    for (final x in v) {
      sumSq += x * x;
    }
    final norm = math.sqrt(sumSq);
    if (norm == 0) return v;
    return v.map((e) => e / norm).toList();
  }

  static Float32List bytesToFloat32(Uint8List blob) {
    if (blob.lengthInBytes % 4 != 0) {
      throw ArgumentError('Invalid embedding blob length ${blob.lengthInBytes} (must be multiple of 4)');
    }

    // sqflite can return Uint8List with a non-zero, non-4-byte-aligned offset.
    final off = blob.offsetInBytes;
    if (off % 4 != 0) {
      // Make a real copy so offset becomes 0 (aligned)
      final copy = Uint8List.fromList(blob);
      return Float32List.view(copy.buffer);
    }

    return Float32List.view(blob.buffer, off, blob.lengthInBytes ~/ 4);
  }

  static double euclideanDistance(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('Embedding dims mismatch: ${a.length} vs ${b.length}');
    }
    double sum = 0.0;
    for (var i = 0; i < a.length; i++) {
      final d = a[i] - b[i];
      sum += d * d;
    }
    return math.sqrt(sum);
  }

  static double cosineSimilarity(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('Embedding dims mismatch: ${a.length} vs ${b.length}');
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (var i = 0; i < a.length; i++) {
      final x = a[i], y = b[i];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    final denom = math.sqrt(na) * math.sqrt(nb);
    if (denom == 0) return 0.0;
    return dot / denom;
  }
}