export default function ForecastPanel({ forecast, livePrice, loading }) {
  if (loading || !forecast) {
    return (
      <div className="panel">
        <h2>Forecast</h2>
        <p className="placeholder">Loading forecast…</p>
      </div>
    )
  }

  const currentPrice = livePrice?.close ?? forecast.last_close
  const isUp = forecast.direction === 'up'

  return (
    <div className="panel">
      <h2>Next-Period Forecast</h2>

      {forecast.data_mode !== 'live' && (
        <div className="mode-note">
          Showing bundled historical demo data — live market data wasn't reachable from this
          environment when this ran.
        </div>
      )}

      <div className={`direction-banner ${isUp ? 'up' : 'down'}`}>
        <span className="arrow">{isUp ? '▲' : '▼'}</span>
        <span className="direction-label">{forecast.direction.toUpperCase()}</span>
        <span className="confidence">{forecast.confidence_pct}% confidence</span>
      </div>

      <div className="row"><span className="label">Current price</span><span className="value mono">{currentPrice}</span></div>
      <div className="row"><span className="label">As of</span><span className="value mono">{forecast.last_date}</span></div>

      <p className="disclaimer-text">{forecast.disclaimer}</p>
    </div>
  )
}
