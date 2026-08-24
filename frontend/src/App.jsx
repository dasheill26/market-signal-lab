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
  const symbolRef = useRef(symbol) // always holds the CURRENT symbol, readable inside
                                    // the WebSocket callback below without that callback
                                    // going stale - see the bug note in the effect itself.
  const prevSymbolRef = useRef(null)

  // Fetch the list of supported symbols once on mount
  useEffect(() => {
    fetch(`${SOCKET_URL}/api/symbols`)
      .then(res => res.json())
      .then(setSymbols)
      .catch(() => setError('Could not reach the backend API.'))
  }, [])

  // Keep symbolRef in sync with the symbol state on every change - this is
  // what makes the WebSocket handler below always compare against the
  // CURRENT symbol, not whatever symbol was selected when the effect first ran.
  useEffect(() => {
    symbolRef.current = symbol
  }, [symbol])

  // Set up the WebSocket connection once. A real bug lived here: this
  // effect has an empty dependency array, so it only runs on mount - the
  // `price_update` handler captured `symbol` by closure at that moment
  // and, since the effect never re-ran, kept comparing every incoming
  // update against that one frozen initial value forever. After a user
  // switched symbols even once, every live update silently failed the
  // `data.symbol === symbol` check and got dropped - which is exactly
  // why the chart stopped visibly updating after the first symbol
  // switch. Fixed by reading symbolRef.current (always fresh) instead
  // of the closed-over `symbol` variable.
  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ['websocket', 'polling'] })
    socketRef.current = socket

    socket.on('price_update', (data) => {
      setLivePrice(prev => (data.symbol === symbolRef.current ? data : prev))
    })

    return () => socket.disconnect()
  }, [])

  // Fetch forecast + chart data whenever the selected symbol or horizon
  // changes, and (re)subscribe the WebSocket to that symbol's live updates -
  // explicitly unsubscribing from the previous symbol first, otherwise the
  // backend keeps polling every symbol ever visited in a session, growing
  // unbounded rather than tracking just the one currently on screen.
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
      if (prevSymbolRef.current && prevSymbolRef.current !== sym) {
        socketRef.current.emit('unsubscribe', { symbol: prevSymbolRef.current })
      }
      socketRef.current.emit('subscribe', { symbol: sym })
      prevSymbolRef.current = sym
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
