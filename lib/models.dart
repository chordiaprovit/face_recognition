import 'dart:convert';
import 'dart:typed_data';

class EnrolledFace {
  final int? id;
  final String label; // e.g., name/userId

  /// Stored in SQLite as BLOB (Float32List bytes).
  /// IMPORTANT: Always store/read as a real Uint8List (offset 0) to avoid alignment errors.
  final Uint8List embeddingBlob;

  final int embeddingDim;
  final int createdAtMs;
  final String? thumbnailPath; // optional local file path
  final Map<String, dynamic>? meta; // JSON-ish (store as TEXT)

  EnrolledFace({
    this.id,
    required this.label,
    required this.embeddingBlob,
    required this.embeddingDim,
    required this.createdAtMs,
    this.thumbnailPath,
    this.meta,
  });

  Map<String, Object?> toMap() => {
        'id': id,
        'label': label,
        'embedding': embeddingBlob, // sqflite will store as BLOB
        'embedding_dim': embeddingDim,
        'created_at_ms': createdAtMs,
        'thumbnail_path': thumbnailPath,
        'meta_json': meta == null ? null : _encodeJson(meta!),
      };

  static EnrolledFace fromMap(Map<String, Object?> map) {
    final raw = map['embedding'];

    // sqflite can return:
    // - Uint8List (sometimes with odd offsetInBytes)
    // - List<int>
    // Normalize to a NEW Uint8List so offsetInBytes == 0 (aligned).
    final Uint8List blob = _toCleanUint8List(raw);

    final dimRaw = map['embedding_dim'];
    final createdRaw = map['created_at_ms'];

    return EnrolledFace(
      id: map['id'] as int?,
      label: map['label'] as String,
      embeddingBlob: blob,
      embeddingDim: dimRaw is int ? dimRaw : (dimRaw as num).toInt(),
      createdAtMs: createdRaw is int ? createdRaw : (createdRaw as num).toInt(),
      thumbnailPath: map['thumbnail_path'] as String?,
      meta: map['meta_json'] == null ? null : _decodeJson(map['meta_json'] as String),
    );
  }

  static Uint8List _toCleanUint8List(Object? raw) {
    if (raw == null) {
      throw StateError('DB row missing embedding blob');
    }

    if (raw is Uint8List) {
      // Force a copy to remove any non-zero / unaligned offsetInBytes.
      return Uint8List.fromList(raw);
    }

    if (raw is List<int>) {
      return Uint8List.fromList(raw);
    }

    throw StateError('Unsupported embedding type from DB: ${raw.runtimeType}');
  }

  static String _encodeJson(Map<String, dynamic> obj) => jsonEncode(obj);

  static Map<String, dynamic> _decodeJson(String s) =>
      (jsonDecode(s) as Map).cast<String, dynamic>();
}

class RecognitionLogEntry {
  final int? id;
  final int? matchedFaceId;
  final String? matchedLabel;
  final double distance;
  final double similarity; // cosine similarity
  final int tsMs;

  RecognitionLogEntry({
    this.id,
    required this.matchedFaceId,
    required this.matchedLabel,
    required this.distance,
    required this.similarity,
    required this.tsMs,
  });

  Map<String, Object?> toMap() => {
        'id': id,
        'matched_face_id': matchedFaceId,
        'matched_label': matchedLabel,
        'distance': distance,
        'similarity': similarity,
        'ts_ms': tsMs,
      };

  static RecognitionLogEntry fromMap(Map<String, Object?> map) => RecognitionLogEntry(
        id: map['id'] as int?,
        matchedFaceId: map['matched_face_id'] as int?,
        matchedLabel: map['matched_label'] as String?,
        distance: (map['distance'] as num).toDouble(),
        similarity: (map['similarity'] as num).toDouble(),
        tsMs: (map['ts_ms'] as num).toInt(),
      );
}