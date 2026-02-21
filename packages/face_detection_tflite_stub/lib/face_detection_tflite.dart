import 'dart:typed_data';

// Minimal stub API matching what's used in the app for tests.

enum FaceDetectionModel { backCamera, frontCamera }

enum FaceDetectionMode { fast, standard, accurate }

class BoundingBox {
  final double left, top, right, bottom;
  BoundingBox({this.left = 0, this.top = 0, this.right = 0, this.bottom = 0});
  double get width => (right - left).abs();
  double get height => (bottom - top).abs();
}

class Face {
  final BoundingBox boundingBox;
  Face(this.boundingBox);
}

class FaceDetector {
  FaceDetector();
  Future<void> initialize({FaceDetectionModel? model}) async {}
  void dispose() {}
  Future<List<Face>> detectFaces(Uint8List imageBytes, {FaceDetectionMode? mode}) async => [];
  List<double> embeddingForFace(Face face, Uint8List imageBytes) => List<double>.filled(192, 0.0);
  // Some versions of the plugin expose `getFaceEmbedding` and/or static helpers.
  Future<List<double>> getFaceEmbedding(Face face, Uint8List imageBytes) async => embeddingForFace(face, imageBytes);

  double compareFaces(List<double> a, List<double> b) => 0.0;
  double faceDistance(List<double> a, List<double> b) => double.infinity;

  // Static wrappers (older API variants)
  static double staticCompareFaces(List<double> a, List<double> b) => FaceDetector().compareFaces(a, b);
  static double staticFaceDistance(List<double> a, List<double> b) => FaceDetector().faceDistance(a, b);
}
