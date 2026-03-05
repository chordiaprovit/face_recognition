import 'dart:typed_data';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'embedding_utils.dart';
import 'face_pipeline.dart';
import 'face_repository.dart';
import 'models.dart';

class EnrollmentService {
  EnrollmentService({
    required FacePipeline pipeline,
    required FaceRepository repo,
  })  : _pipeline = pipeline,
        _repo = repo;

  final FacePipeline _pipeline;
  final FaceRepository _repo;

  /// Enroll a single capture for a label.
  /// Call this multiple times (different angles) for the same label.
  Future<int> enrollFaceFromImage({
    required Uint8List orientedImageBytes,
    required String orientedImagePath,
    required String label,
    Map<String, dynamic>? meta,
    String? thumbnailPath,
    List<Face>? detectedFaces,
  }) async {
    var faces = detectedFaces ??
        await _pipeline.detectFacesFromFile(orientedImagePath);

    if (faces.isEmpty) {
      throw StateError('No face detected. Please retake with a clear face.');
    }

    // Pick largest face
    faces = List.of(faces)
      ..sort((a, b) =>
          (b.boundingBox.width * b.boundingBox.height)
              .compareTo(a.boundingBox.width * a.boundingBox.height));

    final face = faces.first;
    final emb = await _pipeline.embeddingForFace(face, orientedImageBytes);
    final blob = EmbeddingUtils.float32ToBytes(emb);

    final enrolled = EnrolledFace(
      label: label,
      embeddingBlob: blob,
      embeddingDim: emb.length,
      createdAtMs: DateTime.now().millisecondsSinceEpoch,
      thumbnailPath: thumbnailPath,
      meta: meta,
    );
    return _repo.insertFace(enrolled);
  }

  /// Enroll multiple captures for the same label in one call.
  /// [captures] is a list of (orientedImageBytes, orientedImagePath) pairs.
  /// Returns list of inserted row IDs.
  Future<List<int>> enrollMultipleAngles({
    required String label,
    required List<({Uint8List bytes, String path})> captures,
  }) async {
    final ids = <int>[];
    for (final cap in captures) {
      try {
        final id = await enrollFaceFromImage(
          orientedImageBytes: cap.bytes,
          orientedImagePath: cap.path,
          label: label,
        );
        ids.add(id);
      } catch (_) {
        // Skip captures where no face was detected (e.g. bad angle shot)
      }
    }
    if (ids.isEmpty) {
      throw StateError('No faces detected in any of the captures.');
    }
    return ids;
  }
}