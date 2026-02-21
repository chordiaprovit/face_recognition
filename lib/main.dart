import 'dart:typed_data';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'face_pipeline.dart';
import 'face_repository.dart';
import 'enrollment_service.dart';
import 'recognition_service.dart';
import 'widgets/face_preview.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    final cams = await availableCameras();
    runApp(FaceApp(cameras: cams));
  } catch (e) {
    runApp(
      MaterialApp(
        home: Scaffold(
          body: Center(child: Text('Camera init error: $e')),
        ),
      ),
    );
  }
}

class FaceApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const FaceApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FaceHome(cameras: cameras),
    );
  }
}

class FaceHome extends StatefulWidget {
  final List<CameraDescription> cameras;
  const FaceHome({super.key, required this.cameras});

  @override
  State<FaceHome> createState() => _FaceHomeState();
}

class _FaceHomeState extends State<FaceHome> {
  CameraController? _controller;

  final _pipeline = FacePipeline();
  late final FaceRepository _repo;
  late final EnrollmentService _enroll;
  late final RecognitionService _recognize;

  String _status = 'Ready';
  bool _busy = false;

  Uint8List? _lastImageBytes;
  List<Face> _lastFaces = const [];

  @override
  void initState() {
    super.initState();
    _repo = FaceRepository();
    _enroll = EnrollmentService(pipeline: _pipeline, repo: _repo);
    _recognize = RecognitionService(pipeline: _pipeline, repo: _repo);

    _bootstrap();
  }

  Future<void> _bootstrap() async {
    setState(() => _status = 'Requesting permissions...');

    final cam = await Permission.camera.request();
    if (!cam.isGranted) {
      setState(() => _status = 'Camera permission denied');
      return;
    }

    setState(() => _status = 'Initializing pipeline...');
    await _pipeline.initialize();

    // Prefer front camera if available
    final front = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.front);
    final camDesc = front.isNotEmpty ? front.first : widget.cameras.first;

    setState(() => _status = 'Starting camera...');
    final ctrl = CameraController(
      camDesc,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await ctrl.initialize();

    if (!mounted) return;
    setState(() {
      _controller = ctrl;
      _status = 'Ready';
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _pipeline.dispose();
    super.dispose();
  }

  Future<_CaptureResult> _captureOriented() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) {
      throw StateError('Camera not ready');
    }

    final xfile = await ctrl.takePicture();
    final raw = await xfile.readAsBytes();

    // Bake EXIF orientation so ML Kit bbox matches our crop coords
    final orientedBytes = FacePipeline.bakeOrientation(raw);

    // ML Kit file-path detection is most reliable in this setup
    final orientedPath = await FacePipeline.writeTempJpeg(orientedBytes);

    // If selfie mirroring ever looks wrong, you can un-mirror here.
    // final decoded = img.decodeImage(orientedBytes);
    // if (decoded != null && ctrl.description.lensDirection == CameraLensDirection.front) {
    //   final flipped = img.flipHorizontal(decoded);
    //   final flippedBytes = Uint8List.fromList(img.encodeJpg(flipped, quality: 92));
    //   final flippedPath = await FacePipeline.writeTempJpeg(flippedBytes);
    //   return _CaptureResult(path: flippedPath, bytes: flippedBytes);
    // }

    return _CaptureResult(path: orientedPath, bytes: orientedBytes);
  }

  Future<void> _onDetectOnly() async {
    if (_busy) return;

    setState(() {
      _busy = true;
      _status = 'Capturing...';
      _lastFaces = const [];
      _lastImageBytes = null;
    });

    try {
      final cap = await _captureOriented();

      setState(() {
        _lastImageBytes = cap.bytes;
        _status = 'Detecting faces...';
      });

      final faces = await _pipeline.detectFacesFromFile(cap.path);

      setState(() {
        _lastFaces = faces;
        _status = faces.isEmpty ? 'No face detected' : 'Detected ${faces.length} face(s)';
      });
    } catch (e) {
      setState(() => _status = 'Detect error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _onEnroll() async {
    if (_busy) return;

    setState(() {
      _busy = true;
      _status = 'Capturing...';
      _lastFaces = const [];
      _lastImageBytes = null;
    });

    try {
      final cap = await _captureOriented();

      setState(() {
        _lastImageBytes = cap.bytes;
        _status = 'Detecting face...';
      });

      final faces = await _pipeline.detectFacesFromFile(cap.path);
      setState(() => _lastFaces = faces);

      if (faces.isEmpty) {
        setState(() => _status = 'No face detected. Retake with a clear face.');
        return;
      }

      final id = await _enroll.enrollFaceFromImage(
        orientedImageBytes: cap.bytes,
        orientedImagePath: cap.path,
        label: 'Person_${DateTime.now().millisecondsSinceEpoch}',
      );

      setState(() => _status = 'Enrolled ✅  (id=$id)');
    } catch (e) {
      setState(() => _status = 'Enroll error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _onRecognize() async {
    if (_busy) return;

    setState(() {
      _busy = true;
      _status = 'Capturing...';
      _lastFaces = const [];
      _lastImageBytes = null;
    });

    try {
      final cap = await _captureOriented();

      setState(() {
        _lastImageBytes = cap.bytes;
        _status = 'Recognizing...';
      });

      final result = await _recognize.recognizeFromImage(
        orientedImageBytes: cap.bytes,
        orientedImagePath: cap.path,
      );

      // Also show face boxes for debug
      final faces = await _pipeline.detectFacesFromFile(cap.path);
      setState(() => _lastFaces = faces);

      if (!result.matched || result.bestMatch == null) {
        setState(() {
          _status =
              'No match ❌\n'
              'dist: ${result.bestDistance.toStringAsFixed(3)}   '
              'sim: ${result.bestSimilarity.toStringAsFixed(3)}';
        });
        return;
      }

      setState(() {
        _status =
            'Match ✅ ${result.bestMatch!.label}\n'
            'dist: ${result.bestDistance.toStringAsFixed(3)}   '
            'sim: ${result.bestSimilarity.toStringAsFixed(3)}';
      });
    } catch (e) {
      setState(() => _status = 'Recognize error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition (MLKit + MobileFaceNet)'),
      ),
      body: Column(
        children: [
          // ✅ Big live camera region
          Expanded(
            child: ctrl == null || !ctrl.value.isInitialized
                ? Center(child: Text(_status))
                : Stack(
                    children: [
                      Positioned.fill(child: CameraPreview(ctrl)),

                      // ✅ Large status overlay inside camera
                      Positioned(
                        left: 12,
                        right: 12,
                        bottom: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.60),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Text(
                            _status,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                              height: 1.25,
                            ),
                          ),
                        ),
                      ),

                      // ✅ Floating thumbnail (top-right) for last capture + boxes
                      if (_lastImageBytes != null)
                        Positioned(
                          top: 12,
                          right: 12,
                          child: Container(
                            width: 150,
                            height: 150,
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.20),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.white.withOpacity(0.35)),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: FacePreview(
                                imageBytes: _lastImageBytes!,
                                faces: _lastFaces,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
          ),

          // ✅ Bottom controls (small, doesn’t steal camera space)
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _busy ? null : _onDetectOnly,
                    child: const Text('Detect'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _busy ? null : _onEnroll,
                    child: const Text('Enroll'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _busy ? null : _onRecognize,
                    child: const Text('Recognize'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptureResult {
  final String path;
  final Uint8List bytes;
  _CaptureResult({required this.path, required this.bytes});
}