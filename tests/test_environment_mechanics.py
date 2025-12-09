import unittest
import pandas as pd
import numpy as np
from src.strategies.ema_only.rl_env import EmaEnv, RLConfig

class TestEmaEnvMechanics(unittest.TestCase):
    def setUp(self):
        self.cfg = RLConfig()
        self.cfg.fee_pct = 0.0
        self.cfg.slippage_pct = 0.0
        self.cfg.trade_penalty = 0.0
        # Bônus progressivos (combo):
        # B1 (fast vs slow) = 0.05
        # B2 (slow vs ref)  = 0.10
        self.cfg.entry_bonus_fast_over_slow = 0.05
        self.cfg.entry_bonus_full_trend = 0.10
        self.cfg.allow_short = True
        self.cfg.experts_master_enable = False # Disable experts to focus on EMAs for this test
        self.cfg.override_long_gate = True # Bypass gate to test reward logic purely
        self.cfg.override_short_gate = True

    def create_synthetic_data(self, trend='up', length=20):
        dates = pd.date_range(start='2021-01-01', periods=length, freq='1h')
        
        if trend == 'up':
            # Prices increasing: 100, 101, 102...
            close = np.linspace(100, 120, length)
            # EMAs setup for perfect uptrend: Fast > Slow > Ref
            # e.g. Fast=Close-1, Slow=Close-3, Ref=Close-5
            ema_fast = close - 1
            ema_slow = close - 3
            ref_ema = close - 5
        elif trend == 'down':
            # Prices decreasing: 100, 99, 98...
            close = np.linspace(100, 80, length)
            # EMAs setup for perfect downtrend: Fast < Slow < Ref
            # e.g. Fast=Close+1, Slow=Close+3, Ref=Close+5
            ema_fast = close + 1
            ema_slow = close + 3
            ref_ema = close + 5
        else: # flat
            close = np.full(length, 100)
            ema_fast = np.full(length, 100)
            ema_slow = np.full(length, 100)
            ref_ema = np.full(length, 100)

        df = pd.DataFrame({'close': close, 'Date': dates})
        features = pd.DataFrame({
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'ref_ema': ref_ema,
            'atr_rel': np.full(length, 0.01),
            'experts_mean': np.full(length, 0.5), # Neutral
            'exp_trend': np.full(length, 0.5),
            'exp_ref': np.full(length, 0.5)
        })
        return df, features

    def test_perfect_uptrend_rewards(self):
        """Test if Agent gets max bonuses in a perfect uptrend."""
        df, features = self.create_synthetic_data(trend='up')
        env = EmaEnv(df, features, self.cfg)
        env.reset()

        # Action 1: Long
        # Step 0: Open Long
        obs, reward, terminated, truncated, info = env.step(1)
        
        # Expectation:
        # Alignment: Fast > Slow (Level 1) AND Slow > Ref (Level 2)
        # Bônus total de entrada deve ser B1 + B2 = 0.15 (ignorando PnL, que aqui é pequeno).
        self.assertTrue(reward >= 0.15, f"Reward {reward} should include at least 0.15 of alignment bonuses")

    def test_perfect_downtrend_rewards_symmetry(self):
        """Test if Agent gets max bonuses in a perfect downtrend (Symmetry Check)."""
        df, features = self.create_synthetic_data(trend='down')
        env = EmaEnv(df, features, self.cfg)
        env.reset()

        # Action 2: Short
        # Step 0: Open Short
        obs, reward, terminated, truncated, info = env.step(2)

        # Expectation:
        # Alignment: Fast < Slow (Level 1) AND Slow < Ref (Level 2)
        # This confirms the fix: previously it checked Fast < Ref, now Slow < Ref.
        # In this synthetic data: Fast < Slow < Ref is true.
        # So Slow < Ref is True.
        # Bônus total de entrada deve ser B1 + B2 = 0.15.
        self.assertTrue(reward >= 0.15, f"Reward {reward} should include at least 0.15 of alignment bonuses for short symmetry")

    def test_partial_trend_rewards(self):
        """Test partial alignment (Level 1 but not Level 2)."""
        df, features = self.create_synthetic_data(trend='up')
        # Modify features to break Level 2: Slow < Ref
        features['ref_ema'] = features['ema_slow'] + 10 # Ref way above slow -> Downtrend/Correction context
        # So Fast > Slow (Up) but Slow < Ref (Down context) -> Only Level 1 bonus
        
        env = EmaEnv(df, features, self.cfg)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(1)
        
        # Bonus should be Level 1 (0.1) only (plus PnL)
        # PnL is positive (~1.0), so reward > 0.1.
        # To be precise, we can check logic logic by ensuring it's NOT getting the 0.2 bonus purely?
        # Hard to distinguish 0.1 vs 0.2 with PnL noise unless we zero out price change.
        pass 

    def test_rewards_no_price_change(self):
        """Test bonuses with zero price change to isolate reward values."""
        df, features = self.create_synthetic_data(trend='up')
        # Flatten price to remove PnL
        df['close'] = 100.0
        # Keep EMAs aligned for Uptrend
        features['ema_fast'] = 105.0
        features['ema_slow'] = 100.0
        features['ref_ema'] = 95.0
        
        env = EmaEnv(df, features, self.cfg)
        env.reset()
        
        # Open Long
        obs, reward, terminated, truncated, info = env.step(1)
        
        # PnL should be 0.
        # Reward = Bonus (Nível 1 + Nível 2).
        # Fast(105) > Slow(100) -> Level 1
        # Slow(100) > Ref(95) -> Level 2
        # Expected: 0.15 (0.05 + 0.10)
        self.assertAlmostEqual(reward, 0.15, places=4, msg="Reward should be exactly 0.15 (Level 1 + Level 2 Bonus)")

        # Test Short Symmetry with zero price change
        # Fast(95) < Slow(100) < Ref(105)
        features['ema_fast'] = 95.0
        features['ema_slow'] = 100.0
        features['ref_ema'] = 105.0
        env = EmaEnv(df, features, self.cfg)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(2)
        # Expected: 0.15 (0.05 + 0.10)
        self.assertAlmostEqual(reward, 0.15, places=4, msg="Reward should be exactly 0.15 (Short Level 2 Bonus)")

        # Test Short Level 1 only
        # Fast(95) < Slow(100) but Slow(100) > Ref(90) (Ref is lower, so not full downtrend)
        features['ema_fast'] = 95.0
        features['ema_slow'] = 100.0
        features['ref_ema'] = 90.0
        env = EmaEnv(df, features, self.cfg)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(2)
        # Expected: 0.05
        self.assertAlmostEqual(reward, 0.05, places=4, msg="Reward should be exactly 0.05 (Short Level 1 Bonus)")


if __name__ == '__main__':
    unittest.main()
