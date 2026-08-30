import { ReactNode, useState } from "react";

export default function ConfirmDestructive({
  title,
  verify,
  confirmLabel,
  children,
  onClose,
  onConfirm,
}: {
  title: string;
  verify: string;
  confirmLabel: string;
  children: ReactNode;
  onClose: () => void;
  onConfirm: () => void | Promise<void>;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirm() {
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
    } catch (e: any) {
      setError(String(e.message ?? e));
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={() => !busy && onClose()}>
      <div className="card modal destructive-modal" onClick={(e) => e.stopPropagation()}>
        <div className="spread"><b style={{ color: "var(--red)" }}>{title}</b><button onClick={onClose} disabled={busy}>×</button></div>
        <div className="destructive-copy">{children}</div>
        <div className="section-title">type <code>{verify}</code> to confirm</div>
        <input value={typed} onChange={(e) => setTyped(e.target.value)} autoFocus />
        {error && <div className="mono-sm" style={{ color: "var(--red)", marginTop: 8 }}>{error}</div>}
        <div className="row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
          <button onClick={onClose} disabled={busy}>cancel</button>
          <button className="danger-button" onClick={confirm} disabled={typed !== verify || busy}>
            {busy ? "working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
