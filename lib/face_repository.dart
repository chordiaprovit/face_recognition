
import 'package:sqflite/sqflite.dart';
import 'face_db.dart';
import 'models.dart';

class FaceRepository {
  FaceRepository({FaceDb? db}) : _db = db ?? FaceDb.instance;
  final FaceDb _db;

  Future<int> insertFace(EnrolledFace face) async {
    final d = await _db.db;
    return d.insert('faces', face.toMap(), conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<EnrolledFace>> getAllFaces() async {
    final d = await _db.db;
    final rows = await d.query('faces', orderBy: 'created_at_ms DESC');
    return rows.map((r) => EnrolledFace.fromMap(r)).toList();
  }

  Future<int> deleteFace(int id) async {
    final d = await _db.db;
    return d.delete('faces', where: 'id = ?', whereArgs: [id]);
  }

  Future<int> clearFaces() async {
    final d = await _db.db;
    return d.delete('faces');
  }

  Future<int> insertLog(RecognitionLogEntry entry) async {
    final d = await _db.db;
    return d.insert('recognition_log', entry.toMap());
  }

  Future<List<RecognitionLogEntry>> getRecentLogs({int limit = 50}) async {
    final d = await _db.db;
    final rows = await d.query('recognition_log', orderBy: 'ts_ms DESC', limit: limit);
    return rows.map((r) => RecognitionLogEntry.fromMap(r)).toList();
  }
}
