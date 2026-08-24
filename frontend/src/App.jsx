import { useState, useEffect, useCallback, useRef } from 'react'
import { io } from 'socket.io-client'
import SymbolSelector from './components/SymbolSelector.jsx'
import HorizonSelector from './components/HorizonSelector.jsx'
import PriceChart from './components/PriceChart.jsx'
import ForecastPanel from './components/ForecastPanel.jsx'
import BacktestPanel from './components/BacktestPanel.jsx'
import AdvancedAnalysisPanel from './components/AdvancedAnalysisPanel.jsx'
import RiskEducationPanel from './components/RiskEducationPanel.jsx'
import './App.css'

const SOCKET_URL = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [symbols, setSymbols] = useState({ stocks: [], forex: [], metals: [] })
  const [symbol, setSymbol] = useState('NVDA')
  const [horizon, setHorizon] = useState(1)
  const [forecast, setForecast] = useState(null)
  const [chartData, setChartData] = useState([])
  const [livePrice, setLivePrice] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const socketRef = useRef(null)

  // Fetch the list of supported symbols once on mount
  useEffect(() => {
    fetch(`${SOCKET_URL}/api/symbols`)
      .then(res => res.json())
      .then(setSymbols)
      .catch(() => setError('Could not reach the backend API.'))
  }, [])

  // Set up the WebSocket connection once
  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ['websocket', 'polling'] })
    socketRef.current = socket

    socket.on('price_update', (data) => {
      setLivePrice(prev => (data.symbol === symbol ? data : prev))
    })

    return () => socket.disconnect()
  }, [])

  // Fetch forecast + chart data whenever the selected symbol or horizon
  // changes, and (re)subscribe the WebSocket to that symbol's live updates.
  const loadSymbol = useCallback((sym, h) => {
    setLoading(true)
    setError(null)
    setLivePrice(null)
    fetch(`${SOCKET_URL}/api/forecast/${encodeURIComponent(sym)}?horizon=${h}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error)
        setForecast(data.forecast)
        setChartData(data.chart)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))

    if (socketRef.current) {
      socketRef.current.emit('subscribe', { symbol: sym })
    }
  }, [])

  useEffect(() => {
    loadSymbol(symbol, horizon)
  }, [symbol, horizon, loadSymbol])

  return (
    <div className="wrap">
      <header>
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <h1>Market Signal Lab</h1>
            <p className="subtitle">Time-series ML forecasting, honestly backtested</p>
          </div>
        </div>
        <SymbolSelector symbols={symbols} selected={symbol} onSelect={setSymbol} />
      </header>

      <HorizonSelector selected={horizon} onSelect={setHorizon} />

      <div className="disclaimer-banner">
        This is a research/demo tool showing what a properly backtested ML forecast on
        historical technical indicators actually looks like — including how small a real edge
        over a naive baseline genuinely is. <strong>Not financial advice.</strong>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="main-grid">
        <div className="chart-col">
          <PriceChart data={chartData} loading={loading} livePrice={livePrice} />
        </div>
        <div className="side-col">
          <ForecastPanel forecast={forecast} livePrice={livePrice} loading={loading} horizon={horizon} />
          <BacktestPanel forecast={forecast} />
        </div>
      </div>

      <AdvancedAnalysisPanel symbol={symbol} />

      <RiskEducationPanel symbol={symbol} currentPrice={livePrice?.close ?? forecast?.last_close} />

      <footer>
        Built with a gradient-boosted tree model (scikit-learn) on manually implemented
        technical indicators, walk-forward backtested against a naive baseline —
        <a href="https://github.com/dasheill26/market-signal-lab" target="_blank" rel="noreferrer"> source &amp; README</a>
      </footer>
    </div>
  )
}
