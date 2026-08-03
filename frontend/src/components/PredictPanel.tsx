import { useRef, useState } from "react";
import { Spinner } from "@blueprintjs/core";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LABELS, predict, submitFeedback, type Label, type PredictResponse } from "../lib/api";

export function PredictPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const queryClient = useQueryClient();

  const predictMut = useMutation({
    mutationFn: predict,
    onSuccess: (data) => {
      setResult(data);
      setFeedbackSent(false);
      queryClient.invalidateQueries({ queryKey: ["recent"] });
    },
  });

  const feedbackMut = useMutation({
    mutationFn: ({ id, label }: { id: number; label: Label }) => submitFeedback(id, label),
    onSuccess: () => setFeedbackSent(true),
  });

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setFeedbackSent(false);
    predictMut.reset();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  }

  return (
    <div className="card px-6 py-5">
      <div className="eyebrow mb-1">Run inference</div>
      <p className="mb-4 text-[13px] text-ink-soft">
        Upload a 96×96 histopathology patch. The model returns a label and confidence;
        corrections feed the retraining loop.
      </p>

      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
      <div className="flex items-center gap-3">
        <button className="btn-ghost" onClick={() => inputRef.current?.click()}>
          Choose patch
        </button>
        <span className="truncate font-mono text-[12px] text-ink-mute">{file?.name ?? "No file"}</span>
      </div>

      {previewUrl && (
        <img
          src={previewUrl}
          alt="patch preview"
          className="mt-4 h-24 w-24 border border-hairline"
          style={{ imageRendering: "pixelated" }}
        />
      )}

      <div className="mt-4">
        <button
          className="btn-primary"
          disabled={!file || predictMut.isPending}
          onClick={() => file && predictMut.mutate(file)}
        >
          {predictMut.isPending && <Spinner size={14} />}
          Run prediction
        </button>
      </div>

      {predictMut.isError && (
        <p className="mt-4 text-[13px] text-danger">{predictMut.error.message}</p>
      )}

      {result && (
        <div className="mt-6 border-t border-hairline pt-4">
          <div className="eyebrow mb-2">Result</div>
          <div className="flex items-baseline gap-3">
            <span className="text-[22px] font-bold">{result.label}</span>
            <span className="font-mono text-[13px] text-ink-soft">
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </div>

          {result.prediction_id != null ? (
            <div className="mt-4">
              <div className="eyebrow mb-2">Correct label?</div>
              {feedbackSent ? (
                <p className="text-[13px] text-ink-soft">
                  Recorded. Retraining pulls from the feedback table.
                </p>
              ) : (
                <div className="flex gap-2">
                  {LABELS.map((label) => (
                    <button
                      key={label}
                      className="btn-ghost"
                      disabled={feedbackMut.isPending}
                      onClick={() => feedbackMut.mutate({ id: result.prediction_id!, label })}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
              {feedbackMut.isError && (
                <p className="mt-2 text-[13px] text-danger">{feedbackMut.error.message}</p>
              )}
            </div>
          ) : (
            <p className="mt-4 text-[13px] text-ink-mute">
              Prediction logging is off (no database configured), so feedback can't be recorded.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
