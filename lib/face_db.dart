
import 'dart:async';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

class FaceDb {
  FaceDb._();
  static final FaceDb instance = FaceDb._();

  static const _dbName = 'face_db.sqlite';
  static const _dbVersion = 1;

  Database? _db;

  Future<Database> get db async {
    final existing = _db;
    if (existing != null) return existing;
    final databasesPath = await getDatabasesPath();
    final path = p.join(databasesPath, _dbName);
    _db = await openDatabase(
      path,
      version: _dbVersion,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            thumbnail_path TEXT,
            meta_json TEXT
          );
        ''');
        await db.execute('CREATE INDEX idx_faces_label ON faces(label);');

        await db.execute('''
          CREATE TABLE recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matched_face_id INTEGER,
            matched_label TEXT,
            distance REAL NOT NULL,
            similarity REAL NOT NULL,
            ts_ms INTEGER NOT NULL
          );
        ''');
        await db.execute('CREATE INDEX idx_log_ts ON recognition_log(ts_ms);');
      },
    );
    return _db!;
  }

  Future<void> close() async {
    final d = _db;
    _db = null;
    if (d != null) await d.close();
  }
}
