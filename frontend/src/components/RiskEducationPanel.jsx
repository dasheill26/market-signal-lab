import { useState, useEffect } from 'react'

const SOCKET_URL = import.meta.env.VITE_API_URL || ''

export default function RiskEducationPanel({ symbol, currentPrice }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedExample, setSelectedExample] = useState(1) // default to "Common default"
  const [accountSize, setAccountSize] = useState(10000)
  const [riskPct, setRiskPct] = useState(1)
  const [calcResult, setCalcResult] = useState(null)
  const [calcError, setCalcError] = useState(null)

  useEffect(() => {
    setLoading(true)
    fetch(`${SOCKET_URL}/api/risk-education/${encodeURIComponent(symbol)}`)
      .then(res => res.json())
      .then(result => {
        if (!result.error) setData(result)
      })
      .finally(() => setLoading(false))
  }, [symbol])

  useEffect(() => {
    if (!data || !currentPrice) return
    const stopDistance = data.examples[selectedExample].stop_distance
    setCalcError(null)
    fetch(`${SOCKET_URL}/api/position-size`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        account_size: accountSize, risk_pct: riskPct,
        stop_distance: stopDistance, price_per_unit: currentPrice,
      }),
    })
      .then(res => res.json())
      .then(result => {
        if (result.error) { setCalcError(result.error); setCalcResult(null) }
        else { setCalcResult(result); setCalcError(null) }
      })
  }, [data, selectedExample, accountSize, riskPct, currentPrice])

  if (loading || !data) {
    return (
      <div className="panel wide">
        <h2>Risk Management Reference</h2>
        <p className="placeholder">Loading…</p>
      </div>
    )
  }

  return (
    <div className="panel wide">
      <h2>Risk Management Reference (Educational)</h2>
      <p className="risk-intro">
        This is risk-sizing methodology, not a trade signal — it's the same regardless of what
        the forecast above says. <strong>Not financial advice.</strong> ATR (Average True Range)
        measures how much {symbol} typically moves; the examples below are standard reference
        multiples for stop/target distance, not a prescription. Decide the trade yourself — use
        this only to size the risk on it.
      </p>

      <div className="atr-summary">
        <div className="row"><span className="label">Current 14-period ATR</span><span className="value mono">{data.current_atr}</span></div>
        <div className="row"><span className="label">ATR as % of price</span><span className="value mono">{data.atr_pct_of_price}%</span></div>
      </div>

      <table className="risk-table">
        <thead>
          <tr><th></th><th>Label</th><th>Stop distance</th><th>Target distance</th><th>R:R</th></tr>
        </thead>
        <tbody>
          {data.examples.map((ex, i) => (
            <tr key={i} className={selectedExample === i ? 'selected' : ''} onClick={() => setSelectedExample(i)}>
              <td><input type="radio" checked={selectedExample === i} onChange={() => setSelectedExample(i)} /></td>
              <td>{ex.label} ({ex.stop_multiplier}× ATR)</td>
              <td className="mono">{ex.stop_distance}</td>
              <td className="mono">{ex.target_distance}</td>
              <td className="mono">1:{ex.risk_reward_ratio}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="calculator">
        <h3>Position Size Calculator</h3>
        <p className="block-note">Pure arithmetic — you supply every input, nothing here is calculated from the model.</p>
        <div className="calc-inputs">
          <label>
            Account size ($)
            <input type="number" value={accountSize} min="1" onChange={e => setAccountSize(Number(e.target.value))} />
          </label>
          <label>
            Risk per trade (%)
            <input type="number" value={riskPct} min="0.1" max="100" step="0.1" onChange={e => setRiskPct(Number(e.target.value))} />
          </label>
        </div>
        {calcError && <div className="error-banner">{calcError}</div>}
        {calcResult && (
          <div className="calc-results">
            <div className="row"><span className="label">You're risking</span><span className="value mono">${calcResult.risk_amount}</span></div>
            <div className="row"><span className="label">Position size</span><span className="value mono">{calcResult.units} units</span></div>
            <div className="row"><span className="label">Position value</span><span className="value mono">${calcResult.position_value}</span></div>
          </div>
        )}
      </div>
    </div>
  )
}
