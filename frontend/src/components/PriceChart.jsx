import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

export default function PriceChart({ data, loading, livePrice }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#8891a0',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1b2128' },
        horzLines: { color: '#1b2128' },
      },
      rightPriceScale: { borderColor: '#262e37' },
      timeScale: { borderColor: '#262e37' },
      width: containerRef.current.clientWidth,
      height: 420,
    })

    const series = chart.addCandlestickSeries({
      upColor: '#4ade80', downColor: '#f87171',
      borderVisible: false,
      wickUpColor: '#4ade80', wickDownColor: '#f87171',
    })

    chartRef.current = chart
    seriesRef.current = series

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return
    const formatted = data.map(d => ({
      time: d.date, open: d.open, high: d.high, low: d.low, close: d.close,
    }))
    seriesRef.current.setData(formatted)
    chartRef.current?.timeScale().fitContent()
  }, [data])

  // Push a live price update onto the most recent bar without a full reload
  useEffect(() => {
    if (!seriesRef.current || !livePrice || !data || data.length === 0) return
    const lastBar = data[data.length - 1]
    seriesRef.current.update({
      time: lastBar.date,
      open: lastBar.open,
      high: Math.max(lastBar.high, livePrice.close),
      low: Math.min(lastBar.low, livePrice.close),
      close: livePrice.close,
    })
  }, [livePrice, data])

  return (
    <div className="chart-panel">
      <div className="chart-header">
        <h2>Price Chart</h2>
        {loading && <span className="loading-tag">Loading…</span>}
        {livePrice && <span className="live-tag">● live update {livePrice.data_mode === 'live' ? '' : '(demo data)'}</span>}
      </div>
      <div ref={containerRef} className="chart-container" />
    </div>
  )
}
