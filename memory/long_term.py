"""
长期记忆模块 - PostgreSQL持久化存储
企业级偏好管理：置信度计算、冲突解决、生命周期管理
"""
import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# ============================================================
# 偏好类别枚举
# ============================================================
class PreferenceCategory(Enum):
    """偏好类别"""
    FOOD = "food"           # 餐饮偏好
    HOTEL = "hotel"         # 住宿偏好
    TRANSPORT = "transport" # 交通偏好
    SPORTS = "sports"       # 运动偏好
    TIME = "time"           # 时间习惯偏好
    WEATHER = "weather"     # 天气相关偏好
    LOCATION = "location"   # 地点偏好
    GENERAL = "general"     # 通用偏好


# ============================================================
# 偏好来源枚举
# ============================================================
class PreferenceSource(Enum):
    """偏好来源（决定初始置信度）"""
    EXPLICIT = "explicit"     # 用户明确表达："我爱吃辣"
    IMPLICIT = "implicit"     # 隐式推断：从行为推测
    BEHAVIOR = "behavior"     # 直接观察：点了麻辣香锅
    HISTORICAL = "historical" # 历史记录：多次出现


# ============================================================
# 数据结构
# ============================================================
@dataclass
class UserPreference:
    """用户偏好数据结构"""
    user_id: str
    category: str           # PreferenceCategory.value
    key: str                # 偏好键
    value: Any              # 偏好值
    confidence: float = 0.5  # 置信度 0-1
    source: str = "implicit"# PreferenceSource.value
    occurrence_count: int = 1  # 出现次数
    is_explicit: bool = False  # 是否显式表达
    last_updated: datetime = None
    created_at: datetime = None
    expires_at: datetime = None  # 过期时间
    is_active: bool = True       # 是否激活
    metadata: Dict = field(default_factory=dict)  # 元数据（存储context）

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class TravelHistory:
    """行程历史"""
    user_id: str
    destination: str
    start_date: str
    end_date: str = ""
    purpose: str = ""
    preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


# ============================================================
# 置信度计算器
# ============================================================
class ConfidenceCalculator:
    """
    置信度计算器
    基于多个维度计算偏好置信度
    """

    # 基础置信度
    BASE_CONFIDENCE = 0.5

    # 来源加成
    SOURCE_BONUS = {
        PreferenceSource.EXPLICIT.value: 0.35,   # 明确表达
        PreferenceSource.BEHAVIOR.value: 0.20,   # 直接行为
        PreferenceSource.IMPLICIT.value: 0.10,   # 隐式推断
        PreferenceSource.HISTORICAL.value: 0.15, # 历史记录
    }

    # 频次加成阈值
    FREQUENCY_THRESHOLDS = [
        (1, 0.00),   # 1次：无加成
        (3, 0.10),   # 3次：+0.10
        (5, 0.15),   # 5次：+0.15
        (10, 0.20),  # 10次：+0.20
    ]

    # 时间衰减配置
    TIME_DECAY_CONFIG = {
        "half_life_days": 30,      # 半衰期30天
        "max_age_days": 180,        # 最大有效天数180天
        "decay_rate": 0.5,
    }

    # 数据库容量配置
    STORAGE_CONFIG = {
        "max_preferences_per_user": 1000,
        "archive_after_days": 90,
        "compress_after_days": 180,
        "cleanup_batch_size": 100,
    }

    @classmethod
    def calculate(cls, preference: UserPreference, all_user_prefs: List[Dict] = None) -> float:
        """
        综合计算置信度

        公式: confidence = base * (1 + source_bonus) * (1 + frequency_bonus) * time_decay
        """
        # 1. 基础置信度
        confidence = cls.BASE_CONFIDENCE

        # 2. 来源加成
        source_bonus = cls.SOURCE_BONUS.get(preference.source, 0.0)
        if preference.is_explicit:
            source_bonus = max(source_bonus, cls.SOURCE_BONUS[PreferenceSource.EXPLICIT.value])
        confidence += source_bonus

        # 3. 频次加成
        frequency_bonus = cls._get_frequency_bonus(preference.occurrence_count)
        confidence += frequency_bonus

        # 4. 时间衰减
        if preference.last_updated:
            time_decay = cls._calculate_time_decay(preference.last_updated)
            confidence *= time_decay

        # 确保在 [0.1, 1.0] 范围内
        return max(0.1, min(1.0, confidence))

    @classmethod
    def _get_frequency_bonus(cls, occurrence_count: int) -> float:
        """根据出现次数获取加成"""
        bonus = 0.0
        for threshold, bonus_value in cls.FREQUENCY_THRESHOLDS:
            if occurrence_count >= threshold:
                bonus = bonus_value
        return bonus

    @classmethod
    def _calculate_time_decay(cls, last_updated: datetime) -> float:
        """
        计算时间衰减

        使用指数衰减: decay = half_life ^ (-age / half_life)
        age = 当前时间 - last_updated
        """
        age = datetime.now() - last_updated
        age_days = age.total_seconds() / 86400  # 转换为天

        half_life = cls.TIME_DECAY_CONFIG["half_life_days"]
        max_age = cls.TIME_DECAY_CONFIG["max_age_days"]

        # 超过最大有效天数，完全衰减
        if age_days >= max_age:
            return 0.1

        # 指数衰减
        decay = pow(half_life, (-age_days / half_life))
        return max(0.1, min(1.0, decay))

    @classmethod
    def should_update(cls, existing: UserPreference, new: UserPreference) -> Tuple[bool, str]:
        """
        判断是否应该更新现有偏好

        Returns:
            (should_update, reason)
        """
        # 显式偏好优先
        if new.is_explicit and not existing.is_explicit:
            return True, "新偏好是显式表达，现有偏好是隐式推断"

        # 频次更高
        if new.occurrence_count > existing.occurrence_count:
            return True, f"新偏好在{new.occurrence_count}个场景中出现，多于现有的{existing.occurrence_count}个"

        # 时间更新且来源更可靠
        if new.last_updated and existing.last_updated:
            days_diff = (new.last_updated - existing.last_updated).days
            if days_diff > 7 and new.confidence > existing.confidence:
                return True, f"新偏好更新({days_diff}天前)，置信度更高"

        # 相同值但新的是显式
        if new.value == existing.value and new.is_explicit:
            return True, "相同偏好，但新表达更明确"

        return False, "现有偏好更可靠"


# ============================================================
# 偏好冲突解决器
# ============================================================
class PreferenceConflictResolver:
    """
    偏好冲突解决器

    处理同一类别、同一键的不同值冲突
    策略：时间衰减 + 置信度 + 频次
    """

    @classmethod
    def resolve(cls, prefs: List[UserPreference]) -> UserPreference:
        """
        从多个冲突偏好中选择最佳

        选择标准：
        1. 置信度最高
        2. 时间最近
        3. 出现频次最高
        """
        if not prefs:
            return None
        if len(prefs) == 1:
            return prefs[0]

        # 按综合得分排序
        def calc_score(pref: UserPreference) -> float:
            recency_score = cls._recency_score(pref.last_updated)
            return (
                pref.confidence * 0.4 +
                recency_score * 0.3 +
                min(pref.occurrence_count / 10, 1.0) * 0.2 +
                (1.0 if pref.is_explicit else 0.5) * 0.1
            )

        sorted_prefs = sorted(prefs, key=calc_score, reverse=True)
        return sorted_prefs[0]

    @classmethod
    def _recency_score(cls, last_updated: datetime) -> float:
        """计算时间新鲜度得分"""
        if not last_updated:
            return 0.0
        age = datetime.now() - last_updated
        days = age.total_seconds() / 86400
        # 7天内满分，之后递减
        return max(0.0, 1.0 - (days / 30))


# ============================================================
# 存储管理器
# ============================================================
class StorageManager:
    """
    存储管理器

    负责：
    - 数据库容量控制
    - 数据归档
    - 数据压缩
    - 过期数据清理
    """

    def __init__(self, long_term_memory):
        self.memory = long_term_memory
        self.config = ConfidenceCalculator.STORAGE_CONFIG

    def check_and_manage_storage(self, user_id: str) -> Dict[str, Any]:
        """
        检查并管理存储

        1. 检查偏好数量
        2. 归档过期数据
        3. 压缩老旧数据
        4. 清理过期偏好
        """
        result = {
            "archived": 0,
            "compressed": 0,
            "deleted": 0,
            "warnings": []
        }

        try:
            # 1. 检查数量
            count = self._get_preference_count(user_id)
            if count > self.config["max_preferences_per_user"]:
                result["warnings"].append(
                    f"偏好数量({count})超过上限({self.config['max_preferences_per_user']})"
                )
                self._archive_oldest_preferences(user_id, count - self.config["max_preferences_per_user"])

            # 2. 归档90天前的数据
            result["archived"] = self._archive_old_preferences(user_id)

            # 3. 压缩180天前的数据
            result["compressed"] = self._compress_old_preferences(user_id)

            # 4. 清理过期偏好
            result["deleted"] = self._cleanup_expired_preferences(user_id)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _get_preference_count(self, user_id: str) -> int:
        """获取用户偏好数量"""
        conn = self.memory._get_connection()
        if conn is None:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM user_preferences WHERE user_id = %s AND is_active = true",
                (user_id,)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            self.memory._pool.release_connection(conn)
            return count
        except:
            if conn:
                self.memory._pool.release_connection(conn)
            return 0

    def _archive_oldest_preferences(self, user_id: str, count: int) -> int:
        """归档最老的偏好（数量超限时）"""
        conn = self.memory._get_connection()
        if conn is None:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_preferences
                SET is_active = false, metadata = jsonb_set(metadata, '{archived}', 'true')
                WHERE id IN (
                    SELECT id FROM user_preferences
                    WHERE user_id = %s AND is_active = true
                    ORDER BY last_updated ASC
                    LIMIT %s
                )
            """, (user_id, count))
            conn.commit()
            cursor.close()
            self.memory._pool.release_connection(conn)
            return cursor.rowcount
        except:
            if conn:
                self.memory._pool.release_connection(conn)
            return 0

    def _archive_old_preferences(self, user_id: str) -> int:
        """归档90天前的数据"""
        archive_date = datetime.now() - timedelta(days=self.config["archive_after_days"])
        conn = self.memory._get_connection()
        if conn is None:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_preferences
                SET is_active = false,
                    metadata = jsonb_set(metadata, '{archived_at}', %s)
                WHERE user_id = %s
                AND is_active = true
                AND last_updated < %s
            """, (archive_date.isoformat(), user_id, archive_date))
            conn.commit()
            cursor.close()
            self.memory._pool.release_connection(conn)
            return cursor.rowcount
        except:
            if conn:
                self.memory._pool.release_connection(conn)
            return 0

    def _compress_old_preferences(self, user_id: str) -> int:
        """
        压缩180天前的重复偏好为摘要

        将多个相似的偏好记录压缩成一个带频次的摘要记录
        """
        compress_date = datetime.now() - timedelta(days=self.config["compress_after_days"])
        conn = self.memory._get_connection()
        if conn is None:
            return 0

        compressed = 0
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            # 查找180天前有多个相同category+key的记录
            cursor.execute("""
                SELECT category, preference_key, COUNT(*) as cnt
                FROM user_preferences
                WHERE user_id = %s
                AND last_updated < %s
                AND is_active = true
                GROUP BY category, preference_key
                HAVING COUNT(*) > 1
            """, (user_id, compress_date))

            groups = cursor.fetchall()
            cursor.close()

            for group in groups:
                # 保留最新的一条，其他标记为压缩
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_preferences
                    SET is_active = false,
                        metadata = jsonb_set(
                            jsonb_set(metadata, '{compressed}', 'true'),
                            '{compressed_into}',
                            %s::jsonb
                        )
                    WHERE user_id = %s
                    AND category = %s
                    AND preference_key = %s
                    AND is_active = true
                    AND id != (
                        SELECT id FROM user_preferences
                        WHERE user_id = %s AND category = %s AND preference_key = %s
                        ORDER BY last_updated DESC LIMIT 1
                    )
                """, (
                    json.dumps({"count": group["cnt"]}),
                    user_id, group["category"], group["preference_key"],
                    user_id, group["category"], group["preference_key"]
                ))
                compressed += cursor.rowcount

            conn.commit()
            cursor.close()
            self.memory._pool.release_connection(conn)

        except Exception as e:
            if conn:
                self.memory._pool.release_connection(conn)

        return compressed

    def _cleanup_expired_preferences(self, user_id: str) -> int:
        """清理已过期的偏好"""
        conn = self.memory._get_connection()
        if conn is None:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM user_preferences
                WHERE user_id = %s
                AND expires_at IS NOT NULL
                AND expires_at < CURRENT_TIMESTAMP
            """, (user_id,))
            conn.commit()
            cursor.close()
            self.memory._pool.release_connection(conn)
            return cursor.rowcount
        except:
            if conn:
                self.memory._pool.release_connection(conn)
            return 0


# ============================================================
# 偏好提取器 - 从对话中提取偏好
# ============================================================
class PreferenceExtractor:
    """
    从对话中提取偏好信息

    识别模式：
    - 显式表达：我爱/我喜欢/我要
    - 隐式表达：总是/一般/通常
    - 行为表达：（点了/吃了/住了）
    """

    # 显式表达模式
    EXPLICIT_PATTERNS = [
        (r"我(爱|喜欢|想要|要|prefer)\s*(.+)", "explicit"),
        (r"(从来|绝对|一定|必须)\s*(不)?(喜欢|吃|住|坐|用)", "explicit"),
        (r"(不要|不爱|不喜欢|不想|别|不可以).{0,20}(吃|住|坐|用|喝)", "explicit_negative"),
    ]

    # 隐式表达模式
    IMPLICIT_PATTERNS = [
        (r"(一般|通常|经常|往往|总是|usually)\s*(.+)", "implicit"),
        (r"(习惯|偏爱|偏向|更爱)\s*(.+)", "implicit"),
    ]

    # 行为模式
    BEHAVIOR_PATTERNS = [
        (r"\(?(点|订|买|吃了|住了|坐了)\s*(.+)\)?", "behavior"),
    ]

    # 偏好类别关键词映射
    CATEGORY_KEYWORDS = {
        PreferenceCategory.FOOD.value: ["吃", "美食", "餐厅", "菜", "辣", "火锅", "川菜", "粤菜", "早茶", "早餐", "午餐", "晚餐"],
        PreferenceCategory.HOTEL.value: ["酒店", "住宿", "住", "汉庭", "如家", "万豪", "希尔顿", "民宿"],
        PreferenceCategory.TRANSPORT.value: ["高铁", "飞机", "火车", "地铁", "公交", "打车", "开车", "自驾"],
        PreferenceCategory.SPORTS.value: ["篮球", "足球", "跑步", "游泳", "健身", "运动", "打球"],
        PreferenceCategory.TIME.value: ["早上", "上午", "中午", "下午", "晚上", "几点", "时间"],
        PreferenceCategory.WEATHER.value: ["晴天", "雨天", "阴天", "下雨", "晴天", "天气"],
        PreferenceCategory.LOCATION.value: ["北京", "上海", "杭州", "旅游", "出差", "去", "到"],
    }

    @classmethod
    def extract_from_text(cls, text: str, context: Dict = None) -> List[Dict]:
        """
        从文本中提取偏好信息

        Returns:
            List[{
                "category": str,
                "key": str,
                "value": str,
                "source": str,
                "is_explicit": bool,
                "confidence_base": float
            }]
        """
        results = []
        text_lower = text.lower()

        # 1. 检测显式偏好
        for pattern, source_type in cls.EXPLICIT_PATTERNS:
            import re
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pref = cls._parse_match(match, source_type, text, context)
                if pref:
                    results.append(pref)

        # 2. 检测隐式偏好
        for pattern, source_type in cls.IMPLICIT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pref = cls._parse_match(match, source_type, text, context)
                if pref:
                    results.append(pref)

        # 3. 检测行为偏好
        for pattern, source_type in cls.BEHAVIOR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pref = cls._parse_match(match, source_type, text, context)
                if pref:
                    results.append(pref)

        # 4. 基于关键词推断类别
        for pref in results:
            if pref["category"] == PreferenceCategory.GENERAL.value:
                pref["category"] = cls._infer_category(text_lower)

        return results

    @classmethod
    def _parse_match(cls, match: re.Match, source_type: str, text: str, context: Dict) -> Optional[Dict]:
        """解析匹配结果"""
        try:
            full_match = match.group(0)
            value = match.group(1) if match.groups() else full_match

            # 判断正负
            is_negative = "不" in full_match or "别" in full_match or "别" in value

            # 推断类别
            category = cls._infer_category(text.lower())

            # 提取key
            key = cls._infer_key(text.lower(), category)

            return {
                "category": category,
                "key": key,
                "value": value.strip() if not is_negative else f"不{value.strip()}",
                "source": source_type,
                "is_explicit": source_type == "explicit",
                "confidence_base": 0.9 if source_type == "explicit" else 0.6
            }
        except Exception:
            return None

    @classmethod
    def _infer_category(cls, text: str) -> str:
        """根据文本推断偏好类别"""
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return category
        return PreferenceCategory.GENERAL.value

    @classmethod
    def _infer_key(cls, text: str, category: str) -> str:
        """根据类别推断偏好键"""
        key_mapping = {
            PreferenceCategory.FOOD.value: "habit",
            PreferenceCategory.HOTEL.value: "brand",
            PreferenceCategory.TRANSPORT.value: "method",
            PreferenceCategory.SPORTS.value: "activity",
            PreferenceCategory.TIME.value: "time_preference",
            PreferenceCategory.WEATHER.value: "weather_preference",
            PreferenceCategory.LOCATION.value: "destination",
        }
        return key_mapping.get(category, "general")


# ============================================================
# 连接池
# ============================================================
class ConnectionPool:
    """数据库连接池"""

    def __init__(self, config: dict, min_connections: int = 2, max_connections: int = 10):
        self.config = config
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool = None

        if POSTGRES_AVAILABLE:
            self._init_pool()

    def _init_pool(self):
        try:
            self._pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 5432),
                database=self.config.get("database", "multi_agent"),
                user=self.config.get("user", "postgres"),
                password=self.config.get("password", "postgres")
            )
            print("✓ PostgreSQL连接池初始化完成")
        except Exception as e:
            print(f"⚠️ PostgreSQL连接池初始化失败: {e}")
            self._pool = None

    def get_connection(self):
        if self._pool is None:
            return None
        try:
            return self._pool.getconn()
        except Exception:
            return None

    def release_connection(self, conn):
        if self._pool and conn:
            try:
                self._pool.putconn(conn)
            except Exception:
                pass

    def close_all(self):
        if self._pool:
            try:
                self._pool.closeall()
            except Exception:
                pass


# ============================================================
# 长期记忆管理器
# ============================================================
class LongTermMemory:
    """
    长期记忆 - PostgreSQL持久化存储

    企业级特性：
    - 置信度计算
    - 偏好冲突解决
    - 存储生命周期管理
    - 自动归档/压缩/清理
    """

    def __init__(self, connection_config: dict = None):
        self.config = connection_config or {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": os.getenv("PG_PORT", "5432"),
            "database": os.getenv("PG_DATABASE", "multi_agent"),
            "user": os.getenv("PG_USER", "postgres"),
            "password": os.getenv("PG_PASSWORD", "postgres"),
        }
        self._pool = None
        self.storage_manager = None

        if POSTGRES_AVAILABLE:
            try:
                self._pool = ConnectionPool(self.config)
                self.storage_manager = StorageManager(self)
            except Exception as e:
                print(f"⚠️ 连接池初始化失败: {e}")

    def _get_connection(self):
        if self._pool:
            conn = self._pool.get_connection()
            if conn and not conn.closed:
                return conn
        return None

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
                    confidence FLOAT DEFAULT 0.5,
                    source VARCHAR(50) DEFAULT 'implicit',
                    occurrence_count INT DEFAULT 1,
                    is_explicit BOOLEAN DEFAULT false,
                    is_active BOOLEAN DEFAULT true,
                    metadata JSONB DEFAULT '{}',
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, category, preference_key, is_active)
                );

                CREATE TABLE IF NOT EXISTS preference_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    category VARCHAR(100),
                    preference_key VARCHAR(255),
                    old_value TEXT,
                    new_value TEXT,
                    change_reason VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

                CREATE INDEX IF NOT EXISTS idx_prefs_user_active ON user_preferences(user_id, is_active);
                CREATE INDEX IF NOT EXISTS idx_prefs_category ON user_preferences(category);
                CREATE INDEX IF NOT EXISTS idx_prefs_updated ON user_preferences(last_updated);
                CREATE INDEX IF NOT EXISTS idx_travel_user ON travel_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_history_user ON preference_history(user_id);
            """)
            conn.commit()
            cursor.close()
            self._pool.release_connection(conn)
            print("✓ PostgreSQL数据库初始化完成")
            return True
        except Exception as e:
            print(f"⚠️ 数据库初始化失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return False

    def save_preference(self, preference: UserPreference) -> Dict[str, Any]:
        """
        保存用户偏好（带冲突检测和置信度计算）

        Returns:
            {
                "success": bool,
                "action": "created" | "updated" | "conflict_resolved" | "skipped",
                "old_value": Any,
                "new_value": Any,
                "confidence": float,
                "reason": str
            }
        """
        conn = self._get_connection()
        if conn is None:
            return {"success": False, "action": "db_error", "reason": "无法连接数据库"}

        try:
            # 1. 计算置信度
            preference.confidence = ConfidenceCalculator.calculate(preference)

            # 2. 检查是否已存在
            existing = self._get_existing_preference(
                conn, preference.user_id, preference.category, preference.key
            )

            result = {"success": True, "confidence": preference.confidence}

            if existing:
                # 3. 冲突解决
                should_update, reason = ConfidenceCalculator.should_update(
                    existing, preference
                )

                if should_update:
                    result["action"] = "updated"
                    result["reason"] = reason
                    result["old_value"] = existing.preference_value
                    self._update_preference(conn, preference, existing)
                else:
                    result["action"] = "skipped"
                    result["reason"] = reason
                    result["skipped_value"] = preference.value
            else:
                # 4. 新增
                result["action"] = "created"
                result["reason"] = "新偏好"
                self._insert_preference(conn, preference)

            conn.commit()

            # 5. 管理存储
            if self.storage_manager:
                storage_result = self.storage_manager.check_and_manage_storage(preference.user_id)
                result["storage_management"] = storage_result

            cursor = conn.cursor()
            cursor.close()
            self._pool.release_connection(conn)
            return result

        except Exception as e:
            print(f"⚠️ 保存偏好失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return {"success": False, "action": "error", "reason": str(e)}

    def _get_existing_preference(self, conn, user_id: str, category: str, key: str) -> Optional[Dict]:
        """获取现有偏好"""
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM user_preferences
                WHERE user_id = %s AND category = %s AND preference_key = %s AND is_active = true
                ORDER BY last_updated DESC
                LIMIT 1
            """, (user_id, category, key))
            result = cursor.fetchone()
            cursor.close()
            return dict(result) if result else None
        except:
            return None

    def _insert_preference(self, conn, preference: UserPreference):
        """插入新偏好"""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences
                (user_id, category, preference_key, preference_value, confidence,
                 source, occurrence_count, is_explicit, is_active, metadata, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true, %s, CURRENT_TIMESTAMP)
        """, (
            preference.user_id,
            preference.category,
            preference.key,
            json.dumps(preference.value) if isinstance(preference.value, dict) else str(preference.value),
            preference.confidence,
            preference.source,
            preference.occurrence_count,
            preference.is_explicit,
            json.dumps(preference.metadata)
        ))
        cursor.close()

    def _update_preference(self, conn, preference: UserPreference, existing: Dict):
        """更新现有偏好"""
        cursor = conn.cursor()

        # 记录历史
        cursor.execute("""
            INSERT INTO preference_history
                (user_id, category, preference_key, old_value, new_value, change_reason)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            preference.user_id,
            preference.category,
            preference.key,
            existing.get("preference_value"),
            str(preference.value),
            f"更新置信度: {existing.get('confidence')} -> {preference.confidence}"
        ))

        # 更新记录
        cursor.execute("""
            UPDATE user_preferences
            SET preference_value = %s,
                confidence = %s,
                source = %s,
                occurrence_count = %s,
                is_explicit = %s,
                metadata = %s,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = %s AND category = %s AND preference_key = %s AND is_active = true
        """, (
            json.dumps(preference.value) if isinstance(preference.value, dict) else str(preference.value),
            preference.confidence,
            preference.source,
            preference.occurrence_count,
            preference.is_explicit,
            json.dumps(preference.metadata),
            preference.user_id,
            preference.category,
            preference.key
        ))
        cursor.close()

    def get_preferences(self, user_id: str, category: str = None,
                       include_inactive: bool = False) -> List[Dict]:
        """获取用户偏好（自动解决冲突）"""
        conn = self._get_connection()
        if conn is None:
            return self._get_preferences_file(user_id, category)

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if category:
                cursor.execute("""
                    SELECT * FROM user_preferences
                    WHERE user_id = %s AND category = %s AND is_active = true
                    ORDER BY last_updated DESC
                """, (user_id, category))
            else:
                cursor.execute("""
                    SELECT * FROM user_preferences
                    WHERE user_id = %s AND is_active = true
                    ORDER BY category, last_updated DESC
                """, (user_id,))

            results = cursor.fetchall()
            cursor.close()
            self._pool.release_connection(conn)

            # 按 category+key 分组，解决冲突
            grouped = defaultdict(list)
            for row in results:
                key = f"{row['category']}_{row['preference_key']}"
                grouped[key].append(row)

            resolved = []
            for key, prefs in grouped.items():
                if len(prefs) == 1:
                    resolved.append(dict(prefs[0]))
                else:
                    # 转换为 UserPreference 对象进行解决
                    pref_objects = [
                        UserPreference(
                            user_id=p["user_id"],
                            category=p["category"],
                            key=p["preference_key"],
                            value=p["preference_value"],
                            confidence=p["confidence"],
                            source=p["source"],
                            occurrence_count=p["occurrence_count"],
                            is_explicit=p["is_explicit"],
                            last_updated=p["last_updated"]
                        )
                        for p in prefs
                    ]
                    best = PreferenceConflictResolver.resolve(pref_objects)
                    if best:
                        resolved.append({
                            "category": best.category,
                            "preference_key": best.key,
                            "preference_value": best.value,
                            "confidence": best.confidence,
                            "source": best.source,
                            "occurrence_count": best.occurrence_count,
                            "is_explicit": best.is_explicit,
                            "last_updated": best.last_updated
                        })

            return resolved

        except Exception as e:
            print(f"⚠️ 获取偏好失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return self._get_preferences_file(user_id, category)

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
            self._pool.release_connection(conn)
            return True
        except Exception as e:
            print(f"⚠️ 保存行程失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return self._save_travel_file(history)

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
            self._pool.release_connection(conn)
            return [dict(row) for row in results]
        except Exception as e:
            print(f"⚠️ 获取行程失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return self._get_travel_file(user_id, limit)

    def get_preference_history(self, user_id: str, category: str = None,
                               key: str = None, limit: int = 20) -> List[Dict]:
        """获取偏好变更历史"""
        conn = self._get_connection()
        if conn is None:
            return []

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if category and key:
                cursor.execute("""
                    SELECT * FROM preference_history
                    WHERE user_id = %s AND category = %s AND preference_key = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, category, key, limit))
            elif category:
                cursor.execute("""
                    SELECT * FROM preference_history
                    WHERE user_id = %s AND category = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, category, limit))
            else:
                cursor.execute("""
                    SELECT * FROM preference_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))

            results = cursor.fetchall()
            cursor.close()
            self._pool.release_connection(conn)
            return [dict(row) for row in results]
        except Exception as e:
            print(f"⚠️ 获取偏好历史失败: {e}")
            if conn:
                self._pool.release_connection(conn)
            return []

    def close(self):
        if self._pool:
            self._pool.close_all()
            self._pool = None

    # ==================== 文件存储后备方案 ====================

    def _get_file_path(self, user_id: str, table: str = "preferences") -> str:
        base_dir = f"data/memory/{table}"
        os.makedirs(base_dir, exist_ok=True)
        return f"{base_dir}/{user_id}.json"

    def _get_preferences_file(self, user_id: str, category: str = None) -> List[Dict]:
        file_path = self._get_file_path(user_id, "preferences")
        if not os.path.exists(file_path):
            return []

        try:
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
                    "confidence": value.get("confidence", 0.5),
                    "source": value.get("source", "file")
                })
            return results
        except Exception:
            return []

    def _save_travel_file(self, history: TravelHistory) -> bool:
        file_path = self._get_file_path(history.user_id, "travel")
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        data.append({
            "destination": history.destination,
            "start_date": history.start_date,
            "end_date": history.end_date,
            "purpose": history.purpose,
            "preferences": history.preferences,
            "created_at": datetime.now().isoformat()
        })

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _get_travel_file(self, user_id: str, limit: int = 10) -> List[Dict]:
        file_path = self._get_file_path(user_id, "travel")
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[-limit:] if len(data) > limit else data
        except Exception:
            return []
