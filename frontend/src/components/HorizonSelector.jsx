const OPTIONS = [
  { value: 1, label: '1 day', tag: 'Day trader' },
  { value: 3, label: '3 days', tag: 'Short swing' },
  { value: 5, label: '1 week', tag: 'Swing trader' },
  { value: 10, label: '2 weeks', tag: 'Position trader' },
]

export default function HorizonSelector({ selected, onSelect }) {
  return (
    <div className="horizon-selector">
      <span className="group-label">Forecast horizon</span>
      <div className="horizon-pills">
        {OPTIONS.map(opt => (
          <button
            key={opt.value}
            className={`horizon-pill ${selected === opt.value ? 'active' : ''}`}
            onClick={() => onSelect(opt.value)}
            title={opt.tag}
          >
            {opt.label}
            <span className="horizon-tag">{opt.tag}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
