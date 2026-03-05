import 'dart:typed_data';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'embedding_utils.dart';
import 'face_pipeline.dart';
import 'face_repository.dart';
import 'models.dart';

class RecognitionResult {
  final bool matched;
  final EnrolledFace? bestMatch;
  final double bestDistance;
  final double bestSimilarity;

  RecognitionResult({
    required this.matched,
    required this.bestMatch,
    required this.bestDistance,
    required this.bestSimilarity,
  });
}

class RecognitionService {
  RecognitionService({
    required FacePipeline pipeline,
    required FaceRepository repo,
    this.similarityThreshold = 0.65,
    this.distanceThreshold = 0.85,
  })  : _pipeline = pipeline,
        _repo = repo;

  final FacePipeline _pipeline;
  final FaceRepository _repo;

  final double similarityThreshold;
  final double distanceThreshold;

  Future<RecognitionResult> recognizeFromImage({
    required Uint8List orientedImageBytes,
    required String orientedImagePath,
    bool log = true,
  }) async {
    final faces = await _pipeline.detectFacesFromFile(orientedImagePath);

    if (faces.isEmpty) {
      final res = RecognitionResult(
        matched: false,
        bestMatch: null,
        bestDistance: double.infinity,
        bestSimilarity: -1,
      );
      if (log) await _logResult(res);
      return res;
    }

    // Largest face
    final sorted = List.of(faces)
      ..sort((a, b) =>
          (b.boundingBox.width * b.boundingBox.height)
              .compareTo(a.boundingBox.width * a.boundingBox.height));

    final probe = await _pipeline.embeddingForFace(
        sorted.first, orientedImageBytes);
    final probeF32 = Float32List.fromList(probe);

    final enrolled = await _repo.getAllFaces();
    if (enrolled.isEmpty) {
      final res = RecognitionResult(
        matched: false,
        bestMatch: null,
        bestDistance: double.infinity,
        bestSimilarity: -1,
      );
      if (log) await _logResult(res);
      return res;
    }

    // Group embeddings by label — with multi-angle enrollment there are
    // multiple rows per person. We find the BEST matching embedding across
    // ALL rows, then attribute the match to that person's label.
    EnrolledFace? best;
    double bestDist = double.infinity;
    double bestSim = -1.0;

    for (final e in enrolled) {
      final eF32 = EmbeddingUtils.bytesToFloat32(e.embeddingBlob);
      final dist = EmbeddingUtils.euclideanDistance(probeF32, eF32);
      final sim = EmbeddingUtils.cosineSimilarity(probeF32, eF32);

      if (dist < bestDist) {
        bestDist = dist;
        bestSim = sim;
        best = e;
      }
    }

    final matched = best != null &&
        (bestSim >= similarityThreshold || bestDist <= distanceThreshold);

    final result = RecognitionResult(
      matched: matched,
      bestMatch: best,
      bestDistance: bestDist,
      bestSimilarity: bestSim,
    );

    if (log) await _logResult(result);
    return result;
  }

  Future<void> _logResult(RecognitionResult res) async {
    await _repo.insertLog(RecognitionLogEntry(
      matchedFaceId: res.matched ? res.bestMatch?.id : null,
      matchedLabel: res.matched ? res.bestMatch?.label : null,
      distance: res.bestDistance,
      similarity: res.bestSimilarity,
      tsMs: DateTime.now().millisecondsSinceEpoch,
    ));
  }
}