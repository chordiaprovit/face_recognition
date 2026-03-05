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
    runApp(MaterialApp(
      home: Scaffold(body: Center(child: Text('Camera init error: $e'))),
    ));
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

// Guided enrollment steps — shown one at a time above the camera.
const _enrollSteps = [
  'Look straight at the camera',
  'Turn slightly LEFT',
  'Turn slightly RIGHT',
  'Tilt slightly UP',
  'Tilt slightly DOWN',
];

class _FaceHomeState extends State<FaceHome> {
  CameraController? _controller;

  final _pipeline = FacePipeline();
  late final FaceRepository _repo;
  late final EnrollmentService _enroll;
  late final RecognitionService _recognize;

  String _status = 'Ready';
  bool _busy = false;

  // Multi-angle enrollment state
  bool _enrolling = false;
  int _enrollStep = 0;
  String _enrollLabel = '';
  final List<({Uint8List bytes, String path})> _enrollCaptures = [];

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

    final front = widget.cameras.where(
        (c) => c.lensDirection == CameraLensDirection.front);
    final camDesc =
        front.isNotEmpty ? front.first : widget.cameras.first;

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
    final orientedBytes = FacePipeline.bakeOrientation(raw);
    final orientedPath = await FacePipeline.writeTempJpeg(orientedBytes);
    return _CaptureResult(path: orientedPath, bytes: orientedBytes);
  }

  // ─── Single detect ────────────────────────────────────────────────────────

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
      setState(() { _lastImageBytes = cap.bytes; _status = 'Detecting...'; });
      final faces = await _pipeline.detectFacesFromFile(cap.path);
      setState(() {
        _lastFaces = faces;
        _status = faces.isEmpty
            ? 'No face detected'
            : 'Detected ${faces.length} face(s)';
      });
    } catch (e) {
      setState(() => _status = 'Detect error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  // ─── Multi-angle enrollment ───────────────────────────────────────────────

  Future<void> _startEnrollment() async {
    // Ask for the person's name first
    final label = await _askLabel();
    if (label == null || label.trim().isEmpty) return;

    setState(() {
      _enrollLabel = label.trim();
      _enrollCaptures.clear();
      _enrollStep = 0;
      _enrolling = true;
      _status = _enrollSteps[0];
    });
  }

  Future<void> _captureEnrollStep() async {
    if (_busy) return;
    setState(() { _busy = true; _status = 'Capturing...'; });

    try {
      final cap = await _captureOriented();
      final faces = await _pipeline.detectFacesFromFile(cap.path);

      if (faces.isEmpty) {
        setState(() => _status =
            'No face detected — try again.\n${_enrollSteps[_enrollStep]}');
        return;
      }

      _enrollCaptures.add((bytes: cap.bytes, path: cap.path));
      setState(() {
        _lastImageBytes = cap.bytes;
        _lastFaces = faces;
      });

      final next = _enrollStep + 1;
      if (next < _enrollSteps.length) {
        setState(() {
          _enrollStep = next;
          _status = _enrollSteps[next];
        });
      } else {
        // All steps done — commit to DB
        await _finishEnrollment();
      }
    } catch (e) {
      setState(() => _status = 'Capture error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _finishEnrollment() async {
    setState(() => _status = 'Saving ${_enrollCaptures.length} embeddings...');
    try {
      final ids = await _enroll.enrollMultipleAngles(
        label: _enrollLabel,
        captures: _enrollCaptures,
      );
      setState(() {
        _enrolling = false;
        _status =
            'Enrolled ✅  $_enrollLabel\n${ids.length} angle(s) saved';
      });
    } catch (e) {
      setState(() { _enrolling = false; _status = 'Enroll error: $e'; });
    }
  }

  void _cancelEnrollment() {
    setState(() {
      _enrolling = false;
      _enrollCaptures.clear();
      _status = 'Enrollment cancelled';
    });
  }

  // ─── Recognition ─────────────────────────────────────────────────────────

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
      setState(() { _lastImageBytes = cap.bytes; _status = 'Recognizing...'; });

      final result = await _recognize.recognizeFromImage(
        orientedImageBytes: cap.bytes,
        orientedImagePath: cap.path,
      );

      final faces = await _pipeline.detectFacesFromFile(cap.path);
      setState(() => _lastFaces = faces);

      if (!result.matched || result.bestMatch == null) {
        setState(() => _status =
            'No match ❌\n'
            'dist: ${result.bestDistance.toStringAsFixed(3)}  '
            'sim: ${result.bestSimilarity.toStringAsFixed(3)}');
      } else {
        setState(() => _status =
            'Match ✅  ${result.bestMatch!.label}\n'
            'dist: ${result.bestDistance.toStringAsFixed(3)}  '
            'sim: ${result.bestSimilarity.toStringAsFixed(3)}');
      }
    } catch (e) {
      setState(() => _status = 'Recognize error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<String?> _askLabel() async {
    final ctrl = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Enter name'),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          decoration: const InputDecoration(hintText: 'e.g. John'),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, ctrl.text),
              child: const Text('Save')),
        ],
      ),
    );
  }

  // ─── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition'),
      ),
      body: Column(
        children: [
          // Camera + overlays
          Expanded(
            child: ctrl == null || !ctrl.value.isInitialized
                ? Center(child: Text(_status))
                : Stack(
                    children: [
                      Positioned.fill(child: CameraPreview(ctrl)),

                      // Status overlay
                      Positioned(
                        left: 12, right: 12, bottom: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 12),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.60),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Text(
                            _status,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 17,
                              fontWeight: FontWeight.w600,
                              height: 1.3,
                            ),
                          ),
                        ),
                      ),

                      // Enrollment step progress dots
                      if (_enrolling)
                        Positioned(
                          top: 16,
                          left: 0, right: 0,
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: List.generate(_enrollSteps.length, (i) {
                              final done = i < _enrollStep;
                              final current = i == _enrollStep;
                              return AnimatedContainer(
                                duration: const Duration(milliseconds: 250),
                                margin: const EdgeInsets.symmetric(horizontal: 5),
                                width: current ? 18 : 10,
                                height: 10,
                                decoration: BoxDecoration(
                                  color: done
                                      ? Colors.green
                                      : current
                                          ? Colors.white
                                          : Colors.white38,
                                  borderRadius: BorderRadius.circular(5),
                                ),
                              );
                            }),
                          ),
                        ),

                      // Thumbnail preview (top-right)
                      if (_lastImageBytes != null)
                        Positioned(
                          top: 12, right: 12,
                          child: Container(
                            width: 130, height: 130,
                            decoration: BoxDecoration(
                              color: Colors.black26,
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(
                                  color: Colors.white38),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(10),
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

          // Bottom controls
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
            child: _enrolling
                ? Row(
                    children: [
                      Expanded(
                        flex: 3,
                        child: FilledButton.icon(
                          onPressed: _busy ? null : _captureEnrollStep,
                          icon: const Icon(Icons.camera_alt),
                          label: Text(
                              'Capture ${_enrollStep + 1}/${_enrollSteps.length}'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        flex: 1,
                        child: OutlinedButton(
                          onPressed: _cancelEnrollment,
                          child: const Text('Cancel'),
                        ),
                      ),
                    ],
                  )
                : Row(
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
                          onPressed: _busy ? null : _startEnrollment,
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