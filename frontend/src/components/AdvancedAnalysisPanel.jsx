import { useState } from 'react'

const SOCKET_URL = import.meta.env.VITE_API_URL || ''

export default function AdvancedAnalysisPanel({ symbol }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runAnalysis = () => {
    setLoading(true)
    setError(null)
    fetch(`${SOCKET_URL}/api/analysis/${encodeURIComponent(symbol)}`)
      .then(res => res.json())
      .then(result => {
        if (result.error) throw new Error(result.error)
        setData(result)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  return (
    <div className="panel wide">
      <div className="analysis-header">
        <h2>Advanced Analysis</h2>
        <button className="run-btn" onClick={runAnalysis} disabled={loading}>
          {loading ? 'Running… (~15-20s)' : data ? 'Re-run' : 'Run Deep Analysis'}
        </button>
      </div>

      {!data && !loading && (
        <p className="placeholder">
          Compares 3 model families under identical walk-forward validation, tunes
          hyperparameters via time-series cross-validation, computes real permutation-based
          feature importance, and checks whether the model's confidence scores are actually
          trustworthy. Deliberately not run automatically — this takes real compute, on purpose,
          only when you ask for it.
        </p>
      )}

      {error && <div className="error-banner">{error}</div>}

      {data && (
        <div className="analysis-grid">
          <div className="analysis-block">
            <h3>Model Comparison</h3>
            <p className="block-note">Same walk-forward methodology, every model, no exceptions.</p>
            {data.model_comparison.map(m => (
              <div key={m.name} className="model-row">
                <div className="model-row-top">
                  <span className="model-name">{m.name}</span>
                  <span className={`model-acc ${m.beats_baseline ? 'good' : 'neutral'}`}>{m.accuracy_pct}%</span>
                </div>
                <p className="model-desc">{m.description}</p>
              </div>
            ))}
          </div>

          <div className="analysis-block">
            <h3>Hyperparameter Tuning</h3>
            <p className="block-note">RandomizedSearchCV over TimeSeriesSplit (temporally correct — never trains on future data).</p>
            <div className="row"><span className="label">Best CV accuracy</span><span className="value mono">{data.tuning.best_cv_accuracy_pct}%</span></div>
            {Object.entries(data.tuning.best_params).map(([k, v]) => (
              <div key={k} className="row"><span className="label">{k}</span><span className="value mono">{String(v)}</span></div>
            ))}
          </div>

          <div className="analysis-block">
            <h3>Feature Importance</h3>
            <p className="block-note">Permutation importance — measures real predictive impact, not an internal tree heuristic.</p>
            {data.feature_importance.slice(0, 6).map(f => {
              const maxImp = data.feature_importance[0].importance || 1
              const widthPct = Math.max(2, (f.importance / maxImp) * 100)
              return (
                <div key={f.feature} className="importance-row">
                  <span className="importance-label">{f.feature}</span>
                  <div className="importance-track"><div className="importance-fill" style={{ width: `${widthPct}%` }} /></div>
                </div>
              )
            })}
          </div>

          <div className="analysis-block">
            <h3>Calibration Check</h3>
            <p className="block-note">Is a "70% confident" prediction actually right 70% of the time?</p>
            <div className="row">
              <span className="label">Brier score</span>
              <span className={`value mono ${data.calibration.well_calibrated ? 'good-text' : 'bad-text'}`}>
                {data.calibration.brier_score} {data.calibration.well_calibrated ? '(beats 50/50 baseline)' : '(worse than 50/50)'}
              </span>
            </div>
            <p className="calibration-note">{data.calibration.note}</p>
          </div>
        </div>
      )}
    </div>
  )
}
