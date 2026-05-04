"""
长期记忆模块 - PostgreSQL持久化存储
"""
import json
import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class UserPreference:
    """用户偏好数据结构"""
    user_id: str
    category: str  # hotel, airline, food, transport, etc.
    key: str       # preference key
    value: Any     # preference value
    confidence: float = 1.0
    source: str = "conversation"  # conversation, behavior, explicit
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class TravelHistory:
    """行程历史"""
    user_id: str
    destination: str
    start_date: str
    end_date: str
    purpose: str
    preferences: Dict[str, Any]  # 出行偏好
    created_at: datetime = None


class LongTermMemory:
    """
    长期记忆 - PostgreSQL持久化存储
    支持用户偏好、历史行程、行为模式
    """

    def __init__(self, connection_config: dict = None):
        # 默认配置（可以从环境变量或配置读取）
        self.config = connection_config or {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": os.getenv("PG_PORT", "5432"),
            "database": os.getenv("PG_DATABASE", "multi_agent"),
            "user": os.getenv("PG_USER", "postgres"),
            "password": os.getenv("PG_PASSWORD", "postgres"),
        }
        self._conn = None

    def _get_connection(self):
        """获取数据库连接"""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(**self.config)
            except Exception as e:
                print(f"⚠️ PostgreSQL连接失败: {e}, 使用文件存储作为后备")
                return None
        return self._conn

    def init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    preference_key VARCHAR(255) NOT NULL,
                    preference_value TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    source VARCHAR(50) DEFAULT 'conversation',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, category, preference_key)
                );

                CREATE TABLE IF NOT EXISTS travel_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    destination VARCHAR(255) NOT NULL,
                    start_date VARCHAR(50),
                    end_date VARCHAR(50),
                    purpose VARCHAR(255),
                    preferences JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS conversation_summary (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255),
                    summary_text TEXT,
                    key_entities JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX idx_preferences_user ON user_preferences(user_id);
                CREATE INDEX idx_preferences_category ON user_preferences(category);
                CREATE INDEX idx_travel_user ON travel_history(user_id);
            """)
            conn.commit()
            cursor.close()
            print("✓ PostgreSQL数据库初始化完成")
            return True
        except Exception as e:
            print(f"⚠️ 数据库初始化失败: {e}")
            return False

    def save_preference(self, preference: UserPreference) -> bool:
        """保存用户偏好"""
        conn = self._get_connection()
        if conn is None:
            return self._save_preference_file(preference)

        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_preferences
                    (user_id, category, preference_key, preference_value, confidence, source, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, category, preference_key)
                DO UPDATE SET
                    preference_value = EXCLUDED.preference_value,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                preference.user_id,
                preference.category,
                preference.key,
                json.dumps(preference.value) if isinstance(preference.value, dict) else str(preference.value),
                preference.confidence,
                preference.source
            ))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"⚠️ 保存偏好失败: {e}")
            return False

    def get_preferences(self, user_id: str, category: str = None) -> List[Dict]:
        """获取用户偏好"""
        conn = self._get_connection()
        if conn is None:
            return self._get_preferences_file(user_id, category)

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if category:
                cursor.execute("""
                    SELECT * FROM user_preferences
                    WHERE user_id = %s AND category = %s
                    ORDER BY updated_at DESC
                """, (user_id, category))
            else:
                cursor.execute("""
                    SELECT * FROM user_preferences
                    WHERE user_id = %s
                    ORDER BY category, updated_at DESC
                """, (user_id,))

            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"⚠️ 获取偏好失败: {e}")
            return []

    def save_travel_history(self, history: TravelHistory) -> bool:
        """保存行程历史"""
        conn = self._get_connection()
        if conn is None:
            return self._save_travel_file(history)

        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO travel_history
                    (user_id, destination, start_date, end_date, purpose, preferences)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                history.user_id,
                history.destination,
                history.start_date,
                history.end_date,
                history.purpose,
                json.dumps(history.preferences)
            ))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"⚠️ 保存行程失败: {e}")
            return False

    def get_travel_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取行程历史"""
        conn = self._get_connection()
        if conn is None:
            return self._get_travel_file(user_id, limit)

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM travel_history
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (user_id, limit))

            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"⚠️ 获取行程失败: {e}")
            return []

    def save_conversation_summary(self, user_id: str, session_id: str,
                                  summary: str, key_entities: dict = None) -> bool:
        """保存对话摘要"""
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_summary
                    (user_id, session_id, summary_text, key_entities)
                VALUES (%s, %s, %s, %s)
            """, (user_id, session_id, summary, json.dumps(key_entities or {})))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"⚠️ 保存摘要失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ==================== 文件存储后备方案 ====================

    def _get_file_path(self, user_id: str, table: str = "preferences") -> str:
        """获取文件路径"""
        base_dir = f"data/memory/{table}"
        os.makedirs(base_dir, exist_ok=True)
        return f"{base_dir}/{user_id}.json"

    def _save_preference_file(self, preference: UserPreference) -> bool:
        """文件存储后备 - 偏好"""
        file_path = self._get_file_path(preference.user_id, "preferences")
        data = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        key = f"{preference.category}_{preference.key}"
        data[key] = {
            "value": preference.value,
            "confidence": preference.confidence,
            "source": preference.source,
            "updated_at": datetime.now().isoformat()
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def _get_preferences_file(self, user_id: str, category: str = None) -> List[Dict]:
        """文件存储后备 - 获取偏好"""
        file_path = self._get_file_path(user_id, "preferences")
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for key, value in data.items():
            if category and not key.startswith(category):
                continue
            parts = key.split("_", 1)
            results.append({
                "category": parts[0] if len(parts) > 0 else "",
                "preference_key": parts[1] if len(parts) > 1 else key,
                "preference_value": value.get("value"),
                "confidence": value.get("confidence", 1.0),
                "source": value.get("source", "file")
            })
        return results

    def _save_travel_file(self, history: TravelHistory) -> bool:
        """文件存储后备 - 行程"""
        file_path = self._get_file_path(history.user_id, "travel")
        data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        data.append({
            "destination": history.destination,
            "start_date": history.start_date,
            "end_date": history.end_date,
            "purpose": history.purpose,
            "preferences": history.preferences,
            "created_at": datetime.now().isoformat()
        })

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def _get_travel_file(self, user_id: str, limit: int = 10) -> List[Dict]:
        """文件存储后备 - 获取行程"""
        file_path = self._get_file_path(user_id, "travel")
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-limit:]