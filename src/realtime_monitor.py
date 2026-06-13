#!/usr/bin/env python3
"""
涨停低吸策略 — 盘中实时监控程序
每天早上9:30后运行，每分钟拉取持仓和观察中股票的1分钟分时，
根据分时形态判断买入提醒。

【观察中】按原策略买点 -> 突破策略买入价(MA18×1.01)+站稳+站均价+放量

【持仓中】按1分钟K线买卖细则:
  一、均价支撑低吸 — 阶梯回踩均价买点（低点抬高+缩量回踩+均价支撑）
  二、日内反转 — V型反转（深跌2-5%放量突破均价）/ W双底（不创新低二次拉升）
  三、分时突破 — 横盘平台突破（波动<1.5%+放量2倍+突破高点）/ 新高突破
  四、尾盘买入 — 14:40后均价企稳（全天均价支撑+低点抬高）/ 尾盘抢筹拉升
  卖出: 冲高回落 | 跌破均价 | 阶梯下跌 | 量能顶背离 | -5%紧急止损
  🚫 均价线下不出买入信号 | 9:30-9:40不触发 | 同类型信号5分钟不重复
"""
import os, sys, json, time, logging
from datetime import datetime, date, timedelta
import requests
import pandas as pd
import numpy as np

# ====== 配置 ======
PROJ = "/mnt/d/AI-20260604"
SIGNAL_FILE = os.path.join(PROJ, "signals", "zt_pullback_v2_latest.csv")
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
STATE_DIR = os.path.join(PROJ, "alerts")
STATE_FILE = os.path.join(STATE_DIR, "monitor_state.json")
ALERT_LOG = os.path.join(STATE_DIR, "monitor_alerts.log")
STOCK_LIST_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")

API_URL = "https://data.diemeng.chat/api/realtime/history"
API_KEY = "4b4d5c2093ec2260967007116f09a5732e5cbab7f8a17d00da"

OPEN_TIME = "09:30:00"
MIN_WAIT_MINUTES = 15        # 开盘后至少等15分钟才出提醒
STABLE_MINUTES = 3           # 连续几分钟站稳买入价算确认
VOLUME_BOOST = 1.5           # 量能放大倍数阈值
EMERGENCY_DROP = -5.0        # 紧急止损阈值百分比
MAX_RETRIES = 60             # 最多重试次数

os.makedirs(STATE_DIR, exist_ok=True)

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(ALERT_LOG, encoding="utf-8"),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 强制实时刷盘
sys.stdout.reconfigure(line_buffering=True)  # type: ignore


# ============================================================
# 数据加载
# ============================================================
def load_state() -> dict:
    """加载监控状态（已提醒记录等）"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"alerted": {}, "date": str(date.today()), "realtime_buffers": {}}


def save_state(state: dict):
    state["date"] = str(date.today())
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_stock_meta() -> dict:
    """加载股票名称和行业"""
    sl = pd.read_parquet(STOCK_LIST_FILE)
    name_map = dict(zip(sl["ts_code"], sl.get("name", [""] * len(sl))))
    return name_map


def calc_ma18(cls: list) -> float:
    """计算MA18"""
    if len(cls) < 18:
        return None
    return sum(cls[-18:]) / 18


def load_daily_data(code: str) -> pd.DataFrame:
    """加载个股日线"""
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return None
    return pd.read_parquet(fp).sort_values("trade_date")


# ============================================================
# 获取分时数据
# ============================================================
def check_api_permission():
    """检查API权限（先试一把）"""
    try:
        resp = requests.post(
            API_URL,
            headers={"apiKey": API_KEY, "Content-Type": "application/json"},
            json={"stock_code": "601958.SH", "trade_time": datetime.now().strftime("%Y-%m-%d %H:%M:00"), "level": "1min"},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 200:
            logger.info("✅ API权限正常")
            return True
        elif data.get("code") == 402:
            logger.warning(f"❌ API无权限: {data.get('msg')}")
            logger.warning(f"   trace_id: {data.get('trace_id')}")
            return False
        else:
            logger.warning(f"⚠️ API返回异常: {data}")
            return False
    except Exception as e:
        logger.warning(f"❌ API连接失败: {e}")
        return False


def _wait_next_minute():
    """等待到下一分钟的 +5 秒再请求"""
    import time as _time
    now = datetime.now()
    next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    wait_sec = (next_min - now).total_seconds() + 5
    if wait_sec > 0:
        _time.sleep(wait_sec)


def fetch_realtime(codes: list, trade_time: str = None) -> list:
    """
    获取指定股票的1分钟分时数据
    等待到整分钟+5秒再请求，避免空数据重试
    """
    payload = {"level": "1min"}
    if codes:
        payload["stock_code"] = codes if len(codes) > 1 else codes[0]
    if trade_time:
        payload["trade_time"] = trade_time

    for retry in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers={"apiKey": API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 200:
                    items = data.get("data", {}).get("list", [])
                    if items:
                        return items
                    # 空数据：等2秒重试，最多等15秒
                    time.sleep(2)
                    if retry > 7:  # 超过15秒放弃，等下一分钟
                        break
                elif data.get("code") == 402:
                    logger.error(f"API权限不足: {data.get('msg')}")
                    return None
                else:
                    time.sleep(1)
            else:
                logger.warning(f"API返回{resp.status_code}: {resp.text[:100]}")
                time.sleep(1)
        except Exception as e:
            logger.warning(f"请求失败: {e}")
            time.sleep(1)
    return []


def fetch_candle_batch(codes: list, minutes: int = 30) -> dict:
    """
    批量获取持仓股最近n分钟的分时数据
    """
    # 当前时间
    now = datetime.now()
    end_time = now.strftime("%Y-%m-%d %H:%M:00")
    start_time = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:00")

    all_data = {}
    for code in codes:
        payload = {
            "stock_code": code,
            "start_time": start_time,
            "end_time": end_time,
            "level": "1min",
        }
        try:
            resp = requests.post(
                API_URL,
                headers={"apiKey": API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 200:
                    all_data[code] = data["data"].get("list", [])
        except Exception as e:
            logger.warning(f"区间获取{code}失败: {e}")
            time.sleep(0.3)
    return all_data


# ============================================================
# 分时形态分析 — 按1分钟K线买卖细则
# ============================================================

def _calc_vwap(bars: list) -> float:
    """计算分时均价线 VWAP（累积成交额/累积成交量）"""
    total_amount = sum(b.get("amount", 0) or 0 for b in bars)
    total_vol = sum(b.get("vol", 0) or 0 for b in bars)
    return total_amount / total_vol if total_vol > 0 else 0


def _is_above_vwap(price: float, vwap: float) -> bool:
    """多空分界线铁律：价格必须在分时均价线上方"""
    return price >= vwap if vwap > 0 else True


def _reject_alert(code: str, state: dict, alert_type: str, cooldown_minutes: int = 5) -> bool:
    """防重复预警，同一标的同类型买点cooldown分钟内只推送一次"""
    key = f"{code}_{alert_type}"
    last = state.get("alerted", {}).get(key)
    if last:
        last_time = last.get("time", "")
        if last_time:
            try:
                from datetime import datetime
                lt = datetime.strptime(last_time[:19], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - lt).total_seconds() < cooldown_minutes * 60:
                    return True  # 拒绝
            except:
                pass
    return False


def _record_alert(state: dict, code: str, alert_type: str, time: str, price: float):
    state.setdefault("alerted", {})[f"{code}_{alert_type}"] = {"time": time, "price": price}


def _time_filtered(t: str) -> bool:
    """时间过滤：9:30-9:40波动大不触发，11:25-13:00午休暂停"""
    try:
        h, m, s = t[11:].split(":")
        hm = int(h) * 60 + int(m)
        if 9*60+30 <= hm <= 9*60+40:
            return False
        if 11*60+25 <= hm <= 13*60:
            return False
        return True
    except:
        return True


def analyze_realtime(code: str, name: str, entry_price: float, ma18_val: float,
                     recent_bars: list, state: dict, category: str = "watch") -> list:
    """
    分析分时形态，返回提醒列表
    category: "watch"(观察中) / "hold"(持仓中)
    """
    alerts = []

    if not recent_bars or len(recent_bars) < MIN_WAIT_MINUTES:
        return alerts

    if not _time_filtered(recent_bars[-1]["trade_time"]):
        return alerts

    # 解析分时数据
    prices = [b["close"] for b in recent_bars]
    highs = [b["high"] for b in recent_bars]
    lows = [b["low"] for b in recent_bars]
    opens = [b["open"] for b in recent_bars]
    vols = [b.get("vol", 0) or 0 for b in recent_bars]
    amounts = [b.get("amount", 0) or 0 for b in recent_bars]
    volumes = vols
    times = [b["trade_time"] for b in recent_bars]

    # 多空分界线：VWAP
    vwap = _calc_vwap(recent_bars)

    current_price = prices[-1]
    current_high = highs[-1]
    current_low = lows[-1]
    current_time = times[-1]

    # 通用过滤铁律：所有靠谱买点必须在均价线上方
    above_vwap = _is_above_vwap(current_price, vwap)

    # ========================================================
    # 一、观察中 → 首次买入提醒（按原策略买点）
    # ========================================================
    if category == "watch":
        if entry_price and current_price >= entry_price:
            stable_count = 0
            for p in reversed(prices):
                if p >= entry_price:
                    stable_count += 1
                else:
                    break
            if stable_count >= STABLE_MINUTES and above_vwap:
                recent_vols = volumes[-STABLE_MINUTES:]
                past_vols = volumes[-(STABLE_MINUTES + 5):-STABLE_MINUTES] if len(volumes) > STABLE_MINUTES + 5 else volumes[:-(STABLE_MINUTES)]
                avg_past_vol = np.mean(past_vols) if past_vols else 0
                avg_recent_vol = np.mean(recent_vols)
                vol_ok = avg_recent_vol >= avg_past_vol * VOLUME_BOOST if avg_past_vol > 0 else True
                ma_ok = current_price >= ma18_val if ma18_val else True
                if ma_ok and vol_ok:
                    if not _reject_alert(code, state, "buy"):
                        _record_alert(state, code, "buy", current_time, current_price)
                        alerts.append({"type":"BUY","code":code,"name":name,"price":current_price,"entry_price":entry_price,"time":current_time,
                            "msg":f"🟢 买入提醒: {name}({code}) 现价{current_price:.2f}突破买入价{entry_price:.2f}, 站稳{stable_count}分钟, 站均价"})

    # ========================================================
    # 持仓中 → 按1分钟K线买卖细则
    # ========================================================
    if category == "hold":
        if not above_vwap:
            # 跌破均价线不出买入信号，只出卖出信号
            pass

        # 通用前提：价格站稳均价线 + 上涨放量回调缩量
        # 计算最近5分钟均价量和最近5分钟成交量
        recent_5close = prices[-5:] if len(prices) >= 5 else prices
        recent_5vols = volumes[-5:] if len(volumes) >= 5 else volumes
        avg_vol_5 = np.mean(recent_5vols)
        last_3_vols = volumes[-3:] if len(volumes) >= 3 else volumes
        last_3_highs = highs[-3:] if len(highs) >= 3 else highs
        last_3_lows = lows[-3:] if len(lows) >= 3 else lows
        last_3_prices = prices[-3:] if len(prices) >= 3 else prices
        avg_vol_3 = np.mean(last_3_vols)

        # ——————————————————————————————————————
        # 一、均价支撑低吸（阶梯回踩均价线买点）
        # ——————————————————————————————————————
        # 条件：股价在均价线上方，最近3K低点依次抬高，回调缩量，回踩均价未跌破超过1根K
        if above_vwap and len(prices) >= 10:
            lows_3 = lows[-3:]
            # 低点依次抬高
            lows_rising = lows_3[0] <= lows_3[1] <= lows_3[2] if len(lows_3) == 3 else False
            if lows_rising:
                # 回调缩量：近3分钟量 < 整体前段均量
                vol_shrink = avg_vol_3 <= avg_vol_5 * 0.8 if avg_vol_5 > 0 else False
                # 回踩均价未跌破超过1根K：检查最近是否有K跌破，但收盘回到均价上
                broke_vwap = any(b["low"] > vwap * 0.995 and b["close"] < vwap for b in recent_bars[-3:])
                # 当前已确认站稳
                if lows_rising and vol_shrink and not broke_vwap and current_price >= vwap:
                    if not _reject_alert(code, state, "add_step"):
                        _record_alert(state, code, "add_step", current_time, current_price)
                        alerts.append({"type":"ADD_STEP","code":code,"name":name,"price":current_price,"time":current_time,
                            "msg":f"🔵 阶梯回踩加仓: {name}({code}) 均价{vwap:.2f}支撑, 低点抬高, 缩量回调企稳, 现价{current_price:.2f}"})

        # ——————————————————————————————————————
        # 二、日内反转买点（V型/W双底）
        # ——————————————————————————————————————
        if len(prices) >= 20:
            lookback = len(prices)
            min_price = min(prices)
            min_low = min(lows)
            max_price = max(prices)
            drop_pct_from_open = (min(lows[:10]) / prices[0] - 1) * 100 if prices[0] > 0 else 0

            # V型反转：盘中最低跌-2%~-5%，放量拉升突破均价
            if drop_pct_from_open <= -2 and not above_vwap:
                # 当前突破均价
                if current_price >= vwap and vwap > 0 and current_high >= vwap:
                    # 拉升量能检查
                    recent_vol_sum = sum(volumes[-3:])
                    prev_vol_sum = sum(volumes[-8:-3]) if len(volumes) >= 8 else sum(volumes[:5])
                    vol_ratio = recent_vol_sum / prev_vol_sum if prev_vol_sum > 0 else 0
                    if vol_ratio >= 1.5:
                        if not _reject_alert(code, state, "add_v"):
                            _record_alert(state, code, "add_v", current_time, current_price)
                            alerts.append({"type":"ADD_V","code":code,"name":name,"price":current_price,"time":current_time,
                                "msg":f"🔵 V型反转加仓: {name}({code}) 深跌{drop_pct_from_open:.1f}%后放量(×{vol_ratio:.1f})突破均价{vwap:.2f}, 现价{current_price:.2f}"})

            # W双底：两次低点持平，第二个不再创新低，二次拉升站上均价
            if len(prices) >= 30:
                # 找前15分钟低点和后15分钟低点
                first_half_low = min(lows[:len(lows)//2])
                second_half_low = min(lows[len(lows)//2:])
                # 第二个低点不低于第一个的98%（基本持平）
                if second_half_low >= first_half_low * 0.98:
                    # 当前已站上均价
                    if above_vwap:
                        # 二次拉升量能放大
                        last_vol_ratio = volumes[-1] / avg_vol_5 if avg_vol_5 > 0 else 0
                        if last_vol_ratio >= 1.5:
                            if not _reject_alert(code, state, "add_w"):
                                _record_alert(state, code, "add_w", current_time, current_price)
                                alerts.append({"type":"ADD_W","code":code,"name":name,"price":current_price,"time":current_time,
                                    "msg":f"🔵 W双底加仓: {name}({code}) 双底{first_half_low:.2f}/{second_half_low:.2f}不创新低, 二次放量(×{last_vol_ratio:.1f})站均价, 现价{current_price:.2f}"})

        # ——————————————————————————————————————
        # 三、分时突破买点（横盘平台突破/前高突破）
        # ——————————————————————————————————————
        if above_vwap and len(prices) >= 15:
            # 横盘平台突破：近15分钟波动<1.5%，突破横盘高点且量能放大2倍
            recent_15_high = max(highs[-15:])
            recent_15_low = min(lows[-15:])
            recent_15_mid = (recent_15_high + recent_15_low) / 2
            range_pct = (recent_15_high / recent_15_low - 1) * 100 if recent_15_low > 0 else 999

            if range_pct < 1.5:
                # 横盘窄幅震荡
                if current_price >= recent_15_high:
                    # 突破平台上沿
                    vol_ratio = volumes[-1] / (avg_vol_5 + 0.001)
                    if vol_ratio >= 2:
                        if not _reject_alert(code, state, "add_break"):
                            _record_alert(state, code, "add_break", current_time, current_price)
                            alerts.append({"type":"ADD_BREAK","code":code,"name":name,"price":current_price,"time":current_time,
                                "msg":f"🔵 平台突破加仓: {name}({code}) 横盘{range_pct:.1f}%(<1.5%), 放量(×{vol_ratio:.1f})突破{recent_15_high:.2f}, 现价{current_price:.2f}"})

            # 前高突破：刷新日内新高（均价支撑）
            if current_price >= recent_15_high and current_high >= recent_15_high:
                last_vol = volumes[-1]
                past_vol = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else 0
                vol_ok = last_vol >= past_vol * 1.2 if past_vol > 0 else True
                if vol_ok and current_price >= vwap:
                    if not _reject_alert(code, state, "add_new_high", cooldown_minutes=10):
                        _record_alert(state, code, "add_new_high", current_time, current_price)
                        alerts.append({"type":"ADD_NEW_HIGH","code":code,"name":name,"price":current_price,"time":current_time,
                            "msg":f"🔵 新高突破加仓: {name}({code}) 突破日内新高{recent_15_high:.2f}, 均价{vwap:.2f}支撑, 现价{current_price:.2f}"})

        # ——————————————————————————————————————
        # 四、尾盘买入信号（14:40后）
        # ——————————————————————————————————————
        if above_vwap:
            try:
                hh, mm, ss = current_time[11:].split(":")
                cur_min = int(hh) * 60 + int(mm)
                if cur_min >= 14*60+40:
                    # 全天均价支撑，低点抬高
                    if len(prices) >= 30:
                        first_half_low_30 = min(lows[:15]) if len(lows) >= 15 else min(lows)
                        second_half_low_30 = min(lows[-15:]) if len(lows) >= 15 else min(lows)
                        if second_half_low_30 >= first_half_low_30 * 0.99:
                            # 缩量稳在日内高位
                            current_high_pct = (current_price / min(prices) - 1) if min(prices) > 0 else 0
                            if current_price >= vwap:
                                # 尾盘抢筹拉升
                                last_10_vol = sum(volumes[-10:]) if len(volumes) >= 10 else sum(volumes)
                                prev_10_vol = sum(volumes[-20:-10]) if len(volumes) >= 20 else 0
                                if prev_10_vol > 0 and last_10_vol >= prev_10_vol * 1.5:
                                    if not _reject_alert(code, state, "add_tail", cooldown_minutes=10):
                                        _record_alert(state, code, "add_tail", current_time, current_price)
                                        alerts.append({"type":"ADD_TAIL","code":code,"name":name,"price":current_price,"time":current_time,
                                            "msg":f"🔵 尾盘抢筹加仓: {name}({code}) 尾盘10min放量(×{last_10_vol/prev_10_vol:.1f})站均价{vwap:.2f}, 现价{current_price:.2f}"})
            except:
                pass

        # ——————————————————————————————————————
        # 卖出方向
        # ——————————————————————————————————————

        # 条件A：冲高回落（涨超3%后回落）
        first_price = prices[0] if prices else current_price
        day_high = max(highs)
        day_pct_from_first = (current_price / first_price - 1) * 100 if first_price > 0 else 0
        peak_pct = (day_high / first_price - 1) * 100 if first_price > 0 else 0
        drop_from_peak = (day_high - current_price) / day_high * 100 if day_high > 0 else 0

        if peak_pct >= 3 and drop_from_peak >= 1.5:
            if not _reject_alert(code, state, "sell_peak", cooldown_minutes=10):
                _record_alert(state, code, "sell_peak", current_time, current_price)
                alerts.append({"type":"SELL_PEAK","code":code,"name":name,"price":current_price,"time":current_time,
                    "msg":f"🟡 冲高回落: {name}({code}) 最高{day_high:.2f}(+{peak_pct:.1f}%), 回落{current_price:.2f}(-{drop_from_peak:.1f}%), 关注卖出"})

        # 条件B：倒V跌破均价线（绝对不能买入的分时形态）
        if not above_vwap and len(prices) >= 10:
            before_vwap = _is_above_vwap(prices[-5] if len(prices) >= 5 else prices[0], vwap)
            if before_vwap:
                # 之前站均价现在跌破 → 走弱
                if not _reject_alert(code, state, "sell_vwap_break", cooldown_minutes=10):
                    _record_alert(state, code, "sell_vwap_break", current_time, current_price)
                    alerts.append({"type":"SELL_VWAP","code":code,"name":name,"price":current_price,"time":current_time,
                        "msg":f"🟠 跌破均价: {name}({code}) 跌破均价线{vwap:.2f}, 走弱信号, 注意风险"})

        # 条件C：阶梯下跌（反弹碰均价就承压）
        if len(prices) >= 10:
            bounce_count = 0
            for i in range(max(0, len(prices)-10), len(prices)):
                if prices[i] < vwap * 0.995:
                    # 低于均价后反弹
                    if i+1 < len(prices) and prices[i+1] > prices[i] and prices[i+1] < vwap:
                        bounce_count += 1
            if bounce_count >= 3:
                if not _reject_alert(code, state, "sell_stair", cooldown_minutes=15):
                    _record_alert(state, code, "sell_stair", current_time, current_price)
                    alerts.append({"type":"SELL_STAIR","code":code,"name":name,"price":current_price,"time":current_time,
                        "msg":f"🟠 阶梯下跌: {name}({code}) 连续{bounce_count}次反弹碰均价承压, 空头形态, 注意风险"})

        # 条件D：放量拉升后缩量跳水（量能顶背离）
        if len(volumes) >= 6:
            vol_half = len(volumes)//2
            first_half_avg_vol = np.mean(volumes[:vol_half]) if vol_half > 0 else 0
            second_half_avg_vol = np.mean(volumes[vol_half:]) if len(volumes) > vol_half else 0
            if first_half_avg_vol > 0 and second_half_avg_vol < first_half_avg_vol * 0.3:
                # 后半段缩量跳水
                if not _reject_alert(code, state, "sell_diverg", cooldown_minutes=20):
                    _record_alert(state, code, "sell_diverg", current_time, current_price)
                    alerts.append({"type":"SELL_DIVERG","code":code,"name":name,"price":current_price,"time":current_time,
                        "msg":f"🔴 量能顶背离: {name}({code}) 放量拉升后缩量跳水, 量能萎缩至{second_half_avg_vol/first_half_avg_vol*100:.0f}%, 注意风险"})

        # 紧急止损（-5%）
        if entry_price:
            drop_pct = (current_price / entry_price - 1) * 100
            if drop_pct <= EMERGENCY_DROP:
                if not _reject_alert(code, state, "emergency", cooldown_minutes=30):
                    _record_alert(state, code, "emergency", current_time, current_price)
                    alerts.append({"type":"SELL_EMERGENCY","code":code,"name":name,"price":current_price,"entry_price":entry_price,"time":current_time,
                        "msg":f"🔴 紧急止损: {name}({code}) 现价{current_price:.2f} 跌幅{drop_pct:.2f}%，触及-5%止损线!"})

    return alerts


# ============================================================
# 主循环
# ============================================================
def main_loop():
    logger.info("=" * 60)
    logger.info(f"📊 涨停低吸策略 — 盘中监控启动")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API: {API_URL}")
    logger.info("=" * 60)

    # 检查API权限
    if not check_api_permission():
        logger.error("❌ API无权限或连接失败，监控无法启动")
        logger.error("请开通API权限后再运行: https://data.diemeng.chat")
        return

    # 加载数据
    name_map = load_stock_meta()
    state = load_state()
    signals = pd.read_csv(SIGNAL_FILE)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])

    # 取最近3天的信号
    recent_dates = sorted(signals["signal_date"].unique(), reverse=True)[:3]
    recent_sigs = signals[signals["signal_date"].isin(recent_dates)]

    # 构建持仓列表和观察列表
    today_str = date.today().strftime("%Y-%m-%d")
    holdings = []   # (code, 买入价) — 已入场持仓
    watches = []    # (code, 买入价) — 等待买入

    for _, sig in recent_sigs.iterrows():
        code = sig["ts_code"]
        entry = sig["entry_price"]
        sig_date = str(sig["signal_date"])[:10]
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp).sort_values("trade_date")
            last = df.iloc[-1]
            close = float(last["close"])
            high = float(last["high"])
            cls_arr = df["close"].values.astype(np.float64)
            ma18_val = float(np.mean(cls_arr[-18:])) if len(cls_arr) >= 18 else None

            # 当天信号日的股票（今天刚出信号，明天才入场）归入观察中
            # 当天实时监控开盘后是否满足买入条件
            if sig_date == today_str:
                watches.append((code, entry, ma18_val))
            else:
                # 非当天信号：按上日收盘判断持仓还是观察
                buy_cond = high >= entry
                ma_cond = (close >= ma18_val) if ma18_val else False

                if buy_cond and ma_cond:
                    holdings.append((code, entry, ma18_val))
                else:
                    watches.append((code, entry, ma18_val))
        except:
            continue

    logger.info(f"持仓: {len(holdings)}只 | 观察: {len(watches)}只")

    if not holdings and not watches:
        logger.info("无持仓和观察股票，退出")
        return

    for c, e, m in holdings:
        logger.info(f"  📌 {c} {name_map.get(c, '')} 买入价{e:.2f}")
    for c, e, m in watches:
        logger.info(f"  📡 {c} {name_map.get(c, '')} 买入价{e:.2f}")

    # 分钟级监控循环
    monitor_codes = [h[0] for h in holdings] + [w[0] for w in watches]
    
    # 等待到整分钟+5秒，确保首次请求准点发出
    _wait_next_minute()
    name_map_local = name_map
    holdings_map = {h[0]: (h[1], h[2]) for h in holdings}
    watches_map = {w[0]: (w[1], w[2]) for w in watches}

    cycle_count = 0
    now = datetime.now()

    while True:
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            logger.info(f"  ⏳ 等待开盘 (现在{now.strftime('%H:%M')})...")
            time.sleep(60)
            now = datetime.now()
            continue

        if now.hour >= 15:
            logger.info("  📌 收盘已过，监控结束")
            break

        cycle_count += 1
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        # 批量拉取分时
        if cycle_count == 1:
            # 首次批量拉取30分钟历史
            logger.info(f"\n[{ts}] 初次拉取分时数据...")
            candle_data = fetch_candle_batch(monitor_codes, minutes=30)
        else:
            # 后续只拉最新1分钟
            logger.info(f"\n[{ts}] 拉取最新分时...")
            candle_data = {}
            for code in monitor_codes:
                rows = fetch_realtime([code], trade_time=ts)
                if rows:
                    candle_data[code] = rows
                time.sleep(0.3)

        # 分析每只股票
        all_alerts = []
        for code in monitor_codes:
            bars = candle_data.get(code, [])
            if not bars:
                continue

            name = name_map_local.get(code, "")

            if code in watches_map:
                entry, ma18_v = watches_map[code]
                alerts = analyze_realtime(code, name, entry, ma18_v, bars, state, category="watch")

                for a in alerts:
                    state.setdefault("alerted", {})[a.get("type", "buy")] = {
                        "time": a["time"],
                        "price": a["price"],
                    }
                    all_alerts.append(a)
                    logger.info(f"  >>> {a['msg']}")

            if code in holdings_map:
                entry, ma18_v = holdings_map[code]
                alerts = analyze_realtime(code, name, entry, ma18_v, bars, state, category="hold")

                for a in alerts:
                    state.setdefault("alerted", {})[a.get("type", "sell")] = {
                        "time": a["time"],
                        "price": a["price"],
                    }
                    all_alerts.append(a)
                    logger.info(f"  >>> {a['msg']}")

        # 保存状态
        save_state(state)

        # 每20次输出一次状态摘要
        if cycle_count % 20 == 0:
            logger.info(f"  监控运行中 - 第{cycle_count}次轮询 - {len(all_alerts)}个提醒")

        # 等待到下一分钟的第5秒
        _wait_next_minute()
        now = datetime.now()

    logger.info(f"\n✅ 监控结束: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"总轮询: {cycle_count}次")


if __name__ == "__main__":
    main_loop()
