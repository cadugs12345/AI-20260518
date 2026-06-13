"""
因子预警自动化系统
功能：
1. 自动计算所有活跃因子的IC衰减
2. 红色/橙色/绿色三级预警
3. 输出结构化 JSON 报告
4. 依赖：factor_lifecycle.py, factor_engine.py

Usage:
    python alert_system.py                          # 运行一次
    python alert_system.py --watch                  # 持续监控模式
"""
import os, sys, time, json, pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS, LOGS_DIR

# ====== 路径 ======
ALERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts")
os.makedirs(ALERTS_DIR, exist_ok= True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ====== 因子重要性排序 (用于输出顺序) ======
CORE_ORDER = ["60日动量","20日动量","市值","EMA20偏离","120日动量","换手率","EMA5偏离",
              "波动率","ROE","MACD","RSI_24","OBV","EMA10偏离","BOLL位置","量能趋势",
              "RSI_12","RSI_6","净利率","杠杆","利润增速","营收增速","流动性",
              "EMA5","EMA10","EMA20","RSI","高波反转","超跌信号","量价背离","多排强度"]

SKIP_COLS = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","均值","20日收益率",
             "短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值"}


class FactorAlertSystem:
    """因子预警系统"""
    
    def __init__(self, panel_path=None, label="fwd_20d_ret"):
        t0 = time.time()
        if panel_path is None:
            panel_path = os.path.join(DATA_FACTORS, "factor_panel_v5_final.parquet")
        
        print(f"[AlertSystem] 加载数据 {panel_path}...")
        self.panel = pd.read_parquet(panel_path)
        self.panel["trade_date"] = pd.to_datetime(self.panel["trade_date"])
        self.all_dates = sorted(self.panel["trade_date"].unique())
        self.label = label
        
        # 因子列 (排除标签列)
        self.factors = [c for c in self.panel.columns 
                        if c not in SKIP_COLS 
                        and c not in [self.label, 'fwd_5d_ret']
                        and self.panel[c].dtype in ("float64","int64")]
        print(f"[AlertSystem] {len(self.factors)}个因子, {len(self.panel):,}行, "
              f"{self.all_dates[0].date()}~{self.all_dates[-1].date()}, "
              f"用时{time.time()-t0:.1f}s")
        
        # 缓存IC序列（避免重复计算）
        self._ic_cache = {}
    
    def calc_all_ic_series(self, step=20):
        """一次性计算所有核心因子的IC时间序列（高效：只遍历面板一次）"""
        factor_list = [f for f in self.factors if f in self.panel.columns]
        print(f"[AlertSystem] 批量计算{len(factor_list)}个因子的IC序列...")
        
        # 预分配：每个采样日给所有因子一次算完
        from collections import defaultdict
        all_ic = defaultdict(list)
        
        sample_dates = self.all_dates[::step]
        for idx, date in enumerate(sample_dates):
            day = self.panel[self.panel["trade_date"] == date]
            r = day[self.label].values.astype(np.float64)
            r_mask = np.abs(r) < 0.5
            if r_mask.sum() < 50:
                continue
            r_filtered = r[r_mask]
            
            for f in factor_list:
                v = day[f].values.astype(np.float64)
                mask = ~np.isnan(v) & r_mask
                if mask.sum() < 50:
                    continue
                ic, _ = ss.spearmanr(v[mask], r[mask])
                all_ic[f].append({"trade_date": date, "IC": ic})
            
            if (idx+1) % 20 == 0:
                print(f"  [{idx+1}/{len(sample_dates)}] 采样日 {date.date()}")
        
        for f in factor_list:
            self._ic_cache[f] = pd.DataFrame(all_ic.get(f, []))
        print(f"  IC计算完成，缓存{len(factor_list)}个因子")
    
    def calc_ic_series(self, factor_name):
        """获取因子的IC时间序列（从缓存）"""
        return self._ic_cache.get(factor_name, pd.DataFrame())
    
    def analyze_decay(self, factor_name, early_pct=0.5, recent_n=15):
        """
        分析单因子衰减
        early_pct: 早期取前多少比例
        recent_n: 近期取最后N个采样点
        """
        ic_df = self.calc_ic_series(factor_name)
        if ic_df.empty or len(ic_df) < 20:
            return None
        
        early_n = max(int(len(ic_df) * early_pct), 10)
        early = ic_df["IC"].head(early_n)
        recent = ic_df["IC"].tail(min(recent_n, len(ic_df)//2))
        
        early_ir = early.mean() / early.std() if early.std() > 0 else 0
        recent_ir = recent.mean() / recent.std() if recent.std() > 0 else 0
        decay = recent_ir - early_ir
        
        # 近20个采样点的线性趋势
        recent_20 = ic_df.tail(20).copy()
        if len(recent_20) > 10:
            recent_20["idx"] = range(len(recent_20))
            slope, intercept, r_val, p_val, std_err = ss.linregress(recent_20["idx"], recent_20["IC"])
        else:
            slope, p_val = 0, 1
        
        # IC绝对值
        ic_abs = abs(ic_df["IC"].mean())
        recent_ic_mean = recent.mean()
        
        # 判断等级
        if ic_abs < 0.01:
            level = "无效"
            level_code = 0
        elif decay < -0.5:
            level = "严重衰减"
            level_code = 1
        elif decay < -0.3 or (slope < -0.001 and p_val < 0.15):
            level = "需关注"
            level_code = 2
        elif decay < -0.15:
            level = "轻微衰减"
            level_code = 3
        elif decay > 0.3:
            level = "增强"
            level_code = 5
        else:
            level = "正常"
            level_code = 4
        
        return {
            "factor": factor_name,
            "ic_mean": ic_df["IC"].mean(),
            "ic_std": ic_df["IC"].std(),
            "early_ic_ir": early_ir,
            "recent_ic_ir": recent_ir,
            "decay": decay,
            "recent_ic_mean": recent_ic_mean,
            "slope": slope,
            "slope_pval": p_val,
            "level": level,
            "level_code": level_code,
            "trend": "下降" if slope < -0.0005 else ("上升" if slope > 0.0005 else "平稳"),
            "n_points": len(ic_df),
        }
    
    def scan_all(self, max_factors=None):
        """
        扫描因子的IC衰减状态 (只扫核心因子)
        
        Returns:
            pd.DataFrame: 所有因子衰减状态表
            list: 需要预警的因子列表
        """
        # 只扫描核心因子列表
        core_order = ["60日动量","20日动量","市值","EMA20偏离","120日动量","换手率","EMA5偏离",
                      "波动率","ROE","MACD","RSI_24","OBV","EMA10偏离","BOLL位置","量能趋势",
                      "RSI_12","RSI_6","净利率","杠杆","利润增速","营收增速","流动性",
                      "EMA5","EMA10","EMA20","RSI","高波反转","超跌信号","量价背离","多排强度"]
        factor_list = [f for f in core_order if f in self.panel.columns]
        if max_factors:
            factor_list = factor_list[:max_factors]
        
        print(f"[AlertSystem] 扫描{len(factor_list)}个核心因子...")
        
        # 先批量计算IC
        self.calc_all_ic_series()
        
        results = []
        for i, f in enumerate(factor_list):
            result = self.analyze_decay(f)
            if result:
                results.append(result)
            if (i+1) % 5 == 0:
                print(f"  {i+1}/{len(factor_list)}")
        
        df = pd.DataFrame(results)
        
        # 按等级排序（严重优先）
        severity = {1:0, 2:1, 3:2, 0:3, 4:4, 5:5}
        df["_sort_key"] = df["level_code"].map(severity)
        df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
        
        # 预警列表（level_code 1或2）
        alerts = df[df["level_code"].isin([1, 2])].to_dict("records")
        
        return df, alerts
    
    def build_report(self):
        """生成完整预警报告"""
        t0 = time.time()
        print("="*60)
        print("因子预警报告生成")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        df, alerts = self.scan_all()
        
        # ===== 文本摘要 =====
        lines = []
        lines.append(f"因子预警报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"扫描因子: {len(df)} 个")
        lines.append(f"预警因子: {len(alerts)} 个")
        lines.append("")
        
        if alerts:
            lines.append("⚠️ 预警列表:")
            for a in alerts:
                lines.append(f"  [{a['level']}] {a['factor']:16s} | IC均值 {a['ic_mean']*100:+6.2f}% | "
                            f"早期IR {a['early_ic_ir']:+5.2f} → 近期IR {a['recent_ic_ir']:+5.2f} | "
                            f"衰减 {a['decay']:+5.2f}")
        else:
            lines.append("✅ 所有因子状态正常")
        
        lines.append("")
        lines.append("因子状态总览:")
        lines.append(f"{'因子':20s} | {'IC均值':>8s} | {'IC_IR':>7s} | {'早期IR':>7s} | {'近期IR':>7s} | {'衰减':>5s} | {'状态':>6s}")
        lines.append("-" * 75)
        
        for _, r in df.iterrows():
            ir = r["ic_mean"] / r["ic_std"] if r["ic_std"] > 0 else 0
            lines.append(f"{r['factor']:20s} | {r['ic_mean']*100:+6.2f}% | {ir:+6.2f} | "
                        f"{r['early_ic_ir']:+6.2f} | {r['recent_ic_ir']:+6.2f} | "
                        f"{r['decay']:+5.2f} | {r['level']}")
        
        lines.append(f"\n用时: {time.time()-t0:.1f}秒")
        
        text = "\n".join(lines)
        
        # ===== 结构化JSON =====
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_factors": len(df),
            "n_alerts": len(alerts),
            "summary": {
                "status": "🟡 有预警" if alerts else "🟢 正常",
                "message": f"{len(alerts)}个因子需关注" if alerts else "所有因子状态正常",
            },
            "alerts": alerts[:10],  # 只保留前10
            "all_factors": df.to_dict("records"),
        }
        
        return report, text, df
    
    def save_report(self, report, text, df):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # JSON
        json_path = os.path.join(ALERTS_DIR, f"alert_report_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # Text
        txt_path = os.path.join(ALERTS_DIR, f"alert_report_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # CSV
        csv_path = os.path.join(ALERTS_DIR, f"alert_report_{timestamp}.csv")
        df_save = df.copy()
        df_save = df_save.round(4)
        df_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # 最新报告链接
        latest = {"path": json_path, "timestamp": timestamp, "n_alerts": report["n_alerts"]}
        with open(os.path.join(ALERTS_DIR, "latest.json"), "w") as f:
            json.dump(latest, f)
        
        print(f"[AlertSystem] 报告已保存:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")
        print(f"  CSV:  {csv_path}")
        
        return json_path, txt_path, csv_path


# ==================== 主入口 ====================

def run_alert_system(watch=False, interval_hours=24):
    """运行预警系统"""
    system = FactorAlertSystem()
    report, text, df = system.build_report()
    system.save_report(report, text, df)
    
    # 打印摘要
    print(text)
    
    if watch:
        print(f"\n[AlertSystem] 进入监控模式，每{interval_hours}小时检查一次")
        import time as _time
        while True:
            _time.sleep(interval_hours * 3600)
            report, text, df = system.build_report()
            system.save_report(report, text, df)
            print(text)


def check_now_and_exit():
    """运行一次检查并返回预警数量（用于cron）"""
    system = FactorAlertSystem()
    report, text, df = system.build_report()
    system.save_report(report, text, df)
    return report["n_alerts"], report["summary"]["status"]


if __name__ == "__main__":
    if "--watch" in sys.argv:
        run_alert_system(watch=True)
    else:
        run_alert_system()
