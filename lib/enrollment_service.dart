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
    this.maxFacesPerImage = 1,
  })  : _pipeline = pipeline,
        _repo = repo;

  final FacePipeline _pipeline;
  final FaceRepository _repo;

  final int maxFacesPerImage;

  Future<int> enrollFaceFromImage({
    required Uint8List orientedImageBytes,
    required String orientedImagePath,
    required String label,
    Map<String, dynamic>? meta,
    String? thumbnailPath,
    List<Face>? detectedFaces,
  }) async {
    var faces = detectedFaces ?? await _pipeline.detectFacesFromFile(orientedImagePath);

    if (faces.isEmpty) {
      throw StateError('No face detected. Please retake the photo with a clear face.');
    }

    // Largest face
    if (faces.length > maxFacesPerImage) {
      faces = List.of(faces)
        ..sort((a, b) =>
            (b.boundingBox.width * b.boundingBox.height)
                .compareTo(a.boundingBox.width * a.boundingBox.height));
    }

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
}