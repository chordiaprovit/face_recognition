import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FacePreview extends StatefulWidget {
  final Uint8List imageBytes;
  final List<Face> faces;
  const FacePreview({super.key, required this.imageBytes, required this.faces});

  @override
  State<FacePreview> createState() => _FacePreviewState();
}

class _FacePreviewState extends State<FacePreview> {
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _decodeImage();
  }

  Future<void> _decodeImage() async {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(widget.imageBytes, (img) => completer.complete(img));
    final img = await completer.future;
    if (mounted) setState(() => _image = img);
  }

  @override
  Widget build(BuildContext context) {
    if (_image == null) {
      return const SizedBox(
        height: 220,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    final img = _image!;
    return AspectRatio(
      aspectRatio: img.width / img.height,
      child: Stack(
        fit: StackFit.expand,
        children: [
          RawImage(image: img, fit: BoxFit.contain),
          CustomPaint(
            painter: _BoxPainter(
              faces: widget.faces,
              imageWidth: img.width.toDouble(),
              imageHeight: img.height.toDouble(),
            ),
          ),
        ],
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Face> faces;
  final double imageWidth;
  final double imageHeight;
  _BoxPainter({
    required this.faces,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final f in faces) {
      final rect = f.boundingBox;

      final left = rect.left / imageWidth * size.width;
      final top = rect.top / imageHeight * size.height;
      final right = rect.right / imageWidth * size.width;
      final bottom = rect.bottom / imageHeight * size.height;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) =>
      oldDelegate.faces != faces ||
      oldDelegate.imageWidth != imageWidth ||
      oldDelegate.imageHeight != imageHeight;
}