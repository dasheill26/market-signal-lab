export default function BacktestPanel({ forecast }) {
  if (!forecast) return null

  const modelPct = (forecast.backtest_mean_accuracy * 100).toFixed(1)
  const baselinePct = (forecast.backtest_naive_baseline * 100).toFixed(1)
  const edge = (forecast.backtest_mean_accuracy - forecast.backtest_naive_baseline) * 100

  return (
    <div className="panel">
      <h2>Backtest — Is This Actually Any Good?</h2>
      <p className="backtest-intro">
        {forecast.n_backtest_folds} walk-forward folds — trained only on data strictly before
        each test window, never shuffled. This is the honest number, not the best-looking one.
      </p>

      <div className="bar-compare">
        <div className="bar-row">
          <span className="bar-label">Model</span>
          <div className="bar-track"><div className="bar-fill model" style={{ width: `${modelPct}%` }} /></div>
          <span className="bar-value mono">{modelPct}%</span>
        </div>
        <div className="bar-row">
          <span className="bar-label">Naive baseline</span>
          <div className="bar-track"><div className="bar-fill baseline" style={{ width: `${baselinePct}%` }} /></div>
          <span className="bar-value mono">{baselinePct}%</span>
        </div>
      </div>

      <div className={`verdict ${forecast.beats_baseline ? 'good' : 'neutral'}`}>
        {forecast.beats_baseline
          ? `Beats the naive baseline by ${edge.toFixed(1)} points — a small, real edge, not a trading signal.`
          : `Does not beat the naive baseline on this symbol right now — shown honestly, not hidden.`}
      </div>
    </div>
  )
}
