export interface Zone { from: number; to: number; label: string }

interface Props {
  label: string;
  value: number;
  min?: number;
  max: number;
  zone?: Zone;
  zone2?: Zone;
  leftLabel: string;
  rightLabel: string;
  valueText: string;
}

/** A horizontal scale with a marker and an optional shaded zone (e.g. the model's uncertain band). */
export default function Scale({ label, value, min = 0, max, zone, zone2, leftLabel, rightLabel, valueText }: Props) {
  const at = (x: number) => `${Math.min(100, Math.max(0, ((x - min) / (max - min)) * 100))}%`;
  const zoneStyle = (z: Zone) => ({
    left: at(z.from),
    width: `${Math.max(0, parseFloat(at(z.to)) - parseFloat(at(z.from)))}%`,
    backgroundImage:
      "repeating-linear-gradient(135deg, rgba(142,162,255,0.35) 0 4px, rgba(142,162,255,0.08) 4px 8px)",
  });
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm text-muted">{valueText}</span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={Number(value.toFixed(4))}
        aria-valuetext={valueText}
        className="relative mt-2 h-3 rounded-full bg-raised"
      >
        {[zone, zone2].map(
          (z) => z && <div key={z.label} className="absolute inset-y-0 rounded-sm" style={zoneStyle(z)} title={z.label} />,
        )}
        <div className="absolute -top-1 h-5 w-1 -translate-x-1/2 rounded-full bg-model" style={{ left: at(value) }} />
      </div>
      <div className="mt-1.5 flex justify-between text-xs text-muted">
        <span>{leftLabel}</span>
        {zone && <span className="text-model/90">hatched: {zone.label}</span>}
        <span>{rightLabel}</span>
      </div>
    </div>
  );
}
