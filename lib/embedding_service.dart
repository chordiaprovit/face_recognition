import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'embedding_utils.dart';

class EmbeddingService {
  EmbeddingService._(this._interpreter, this.inputSize, this.embeddingDim);

  final Interpreter _interpreter;
  final int inputSize;      // e.g. 112
  final int embeddingDim;   // e.g. 192

  static EmbeddingService? _instance;

  static Future<EmbeddingService> init({
    String modelAssetPath = 'assets/models/mobilefacenet_112x112.tflite',
    int inputSize = 112,
    int embeddingDim = 192,
    int threads = 2,
  }) async {
    if (_instance != null) return _instance!;
    final opts = InterpreterOptions()..threads = threads;
    final interpreter = await Interpreter.fromAsset(modelAssetPath, options: opts);
    _instance = EmbeddingService._(interpreter, inputSize, embeddingDim);
    return _instance!;
  }

  static EmbeddingService get instance {
    final inst = _instance;
    if (inst == null) {
      throw StateError('EmbeddingService not initialized. Call EmbeddingService.init() first.');
    }
    return inst;
  }

  void close() {
    _interpreter.close();
    _instance = null;
  }

  List<double> embedFromCroppedFace(img.Image faceRgb) {
    final resized = img.copyResize(faceRgb, width: inputSize, height: inputSize);

    final input = _imageToFloat32Input(resized);
    final output = List.filled(embeddingDim, 0.0).reshape([1, embeddingDim]);

    _interpreter.run(input, output);

    final raw = (output[0] as List<double>);
    return EmbeddingUtils.l2Normalize(raw);
  }

  List<List<List<List<double>>>> _imageToFloat32Input(img.Image image) {
    final h = image.height;
    final w = image.width;

    final data = List.generate(1, (_) {
      return List.generate(h, (y) {
        return List.generate(w, (x) {
          final p = image.getPixel(x, y);
          final r = p.r;
          final g = p.g;
          final b = p.b;

          // 0..255 -> [-1, 1]
          final rf = (r - 127.5) / 127.5;
          final gf = (g - 127.5) / 127.5;
          final bf = (b - 127.5) / 127.5;
          return <double>[rf, gf, bf];
        });
      });
    });

    return data;
  }

  static img.Image decodeJpeg(Uint8List jpegBytes) {
    final decoded = img.decodeImage(jpegBytes);
    if (decoded == null) throw ArgumentError('Could not decode image bytes');
    return decoded;
  }
}

extension _Reshape on List<double> {
  List<List<double>> reshape(List<int> shape) {
    if (shape.length != 2 || shape[0] != 1 || shape[1] != length) {
      throw ArgumentError('Unsupported reshape: $shape for length=$length');
    }
    return [this];
  }
}