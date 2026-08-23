export default function SymbolSelector({ symbols, selected, onSelect }) {
  return (
    <div className="symbol-selector">
      <div className="symbol-group">
        <span className="group-label">Stocks</span>
        <div className="symbol-pills">
          {symbols.stocks?.map(s => (
            <button
              key={s}
              className={`pill ${selected === s ? 'active' : ''}`}
              onClick={() => onSelect(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="symbol-group">
        <span className="group-label">Forex</span>
        <div className="symbol-pills">
          {symbols.forex?.map(s => (
            <button
              key={s}
              className={`pill ${selected === s ? 'active' : ''}`}
              onClick={() => onSelect(s)}
            >
              {s.replace('=X', '')}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
