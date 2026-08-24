const CATEGORY_LABELS = { stocks: 'Stocks', forex: 'Forex', metals: 'Metals' }

export default function SymbolSelector({ symbols, selected, onSelect }) {
  const categories = Object.keys(symbols).filter(cat => symbols[cat]?.length > 0)

  return (
    <div className="symbol-selector">
      {categories.map(category => (
        <div className="symbol-group" key={category}>
          <span className="group-label">{CATEGORY_LABELS[category] || category}</span>
          <div className="symbol-pills">
            {symbols[category].map(s => (
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
      ))}
    </div>
  )
}
