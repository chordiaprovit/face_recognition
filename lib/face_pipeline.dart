import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'embedding_service.dart';

class FacePipeline {
  FacePipeline();

  FaceDetector? _detector;
  bool _initialized = false;

  /// Serializing lock — prevents concurrent inference calls.
  Future<void> _lastOp = Future.value();

  Future<T> _runLocked<T>(Future<T> Function() fn) {
    final next = _lastOp.catchError((_) {}).then((_) => fn());
    _lastOp = next.then((_) {}).catchError((_) {});
    return next;
  }

  Future<void> initialize() async {
    if (_initialized) return;

    _detector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.fast,
        enableLandmarks: true,
        enableTracking: true,
      ),
    );

    // Init embedding model once.
    await EmbeddingService.init(
      modelAssetPath: 'assets/models/arcface_mobilefacenet.tflite',
      inputSize: 112,   // ArcFace MobileFaceNet still uses 112x112
      embeddingDim: 512,
      threads: 2,
    );

    _initialized = true;
  }

  Future<void> dispose() async {
    if (!_initialized) return;
    _initialized = false;
    await _runLocked(() async {
      await _detector?.close();
      _detector = null;
    });
  }

  Future<List<Face>> detectFacesFromFile(String filePath) async {
    if (!_initialized) throw StateError('FacePipeline not initialized');
    final detector = _detector!;
    return _runLocked(() async {
      final input = InputImage.fromFilePath(filePath);
      return detector.processImage(input);
    });
  }

  /// Compute embedding from detected face + the SAME oriented JPEG bytes.
  Future<List<double>> embeddingForFace(Face face, Uint8List orientedJpegBytes) async {
    if (!_initialized) throw StateError('FacePipeline not initialized');

    return _runLocked(() async {
      final full = EmbeddingService.decodeJpeg(orientedJpegBytes);

      final rect = face.boundingBox; // Rect in pixel coordinates
      // add a bit of padding
      final padX = (rect.width * 0.15).round();
      final padY = (rect.height * 0.20).round();

      final x = (rect.left.round() - padX).clamp(0, full.width - 1);
      final y = (rect.top.round() - padY).clamp(0, full.height - 1);
      final r = (rect.right.round() + padX).clamp(0, full.width);
      final b = (rect.bottom.round() + padY).clamp(0, full.height);

      final w = (r - x).clamp(1, full.width - x);
      final h = (b - y).clamp(1, full.height - y);

      final cropped = img.copyCrop(full, x: x, y: y, width: w, height: h);

      return EmbeddingService.instance.embedFromCroppedFace(cropped);
    });
  }

  /// Utility: write bytes to a temp file and return path
  static Future<String> writeTempJpeg(Uint8List bytes) async {
    final path = '${Directory.systemTemp.path}/face_${DateTime.now().millisecondsSinceEpoch}.jpg';
    await File(path).writeAsBytes(bytes, flush: true);
    return path;
  }

  /// Make image upright (EXIF) so MLKit bbox + crop match.
  static Uint8List bakeOrientation(Uint8List rawJpeg) {
    final decoded = img.decodeImage(rawJpeg);
    if (decoded == null) return rawJpeg;
    final oriented = img.bakeOrientation(decoded);
    return Uint8List.fromList(img.encodeJpg(oriented, quality: 92));
  }
}