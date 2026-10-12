import { useState } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, AlertTriangle, ArrowRight, ShieldCheck, FileText } from 'lucide-react';
import './UploadPage.css';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [streamProgress, setStreamProgress] = useState(null);
  
  // Single transaction manual entry state
  const [amount, setAmount] = useState('150.00');
  const [time, setTime] = useState('3600');
  const [singleResult, setSingleResult] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleBatchUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setStreamProgress(null);
    setResults(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/transactions/score/batch/stream', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '');
            try {
              const data = JSON.parse(dataStr);
              setStreamProgress(data);
              
              if (data.status === 'Complete!') {
                 setResults({ total: data.total, flagged: data.flagged });
                 setIsUploading(false);
              }
            } catch (e) {
              console.error('Failed to parse SSE JSON', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Batch upload failed', error);
      alert('Upload failed: ' + error.message);
      setIsUploading(false);
    }
  };

  const handleSingleScore = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    
    try {
      // Build dummy payload for V1-V28, we'll just set V14, V17 to some fraud-like values 
      // if amount > 500 just for demo interaction, else 0
      const isSimulatedFraud = parseFloat(amount) > 1000;
      
      const payload = {
        time: parseFloat(time),
        amount: parseFloat(amount),
        v17: isSimulatedFraud ? -2.5 : 0.1,
        v14: isSimulatedFraud ? -3.0 : 0.05,
        v12: isSimulatedFraud ? -2.8 : 0.1,
      };
      
      const response = await axios.post('/transactions/score', payload);
      setSingleResult(response.data);
    } catch (error) {
      console.error('Scoring failed', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-container animate-fade-in">
      <header className="page-header">
        <div>
          <h2>Score Transactions</h2>
          <p className="text-muted">Upload a batch CSV or score a single transaction in real-time.</p>
        </div>
      </header>

      <div className="scoring-grid">
        {/* Batch Upload Section */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <FileText color="var(--gold)" />
            <h3>Batch CSV Scoring</h3>
          </div>
          
          <div className="upload-dropzone">
            <UploadCloud size={48} className="text-muted mb-4" />
            <p className="mb-2">Drag and drop a CSV file here, or click to browse.</p>
            <p className="text-xs text-muted mb-4">Must contain Time, Amount, and V1-V28 columns.</p>
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileChange} 
              className="file-input"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="btn btn-secondary">
              Select File
            </label>
            {file && <p className="mt-2 text-sm text-success font-bold">Selected: {file.name}</p>}
          </div>

          <button 
            className="btn btn-primary w-full mt-4" 
            onClick={handleBatchUpload}
            disabled={!file || isUploading}
          >
            {isUploading ? 'Processing...' : 'Run Batch Score'}
          </button>

          {streamProgress && !results && (
            <div className="stream-progress mt-4 animate-fade-in p-4 glass-card border border-light">
              <h4 className="text-sm font-bold text-muted uppercase tracking-wider mb-2">Live Progress</h4>
              <p className="text-md mb-2">{streamProgress.status}</p>
              
              <div className="w-full bg-[var(--dark-muted)] rounded-full h-2.5 mb-2 overflow-hidden">
                <div 
                  className="bg-[var(--gold)] h-2.5 rounded-full transition-all duration-300" 
                  style={{ width: `${streamProgress.progress}%` }}
                ></div>
              </div>
              <p className="text-xs text-right text-muted">{streamProgress.progress}%</p>
            </div>
          )}

          {results && (
            <div className="batch-results mt-4 animate-fade-in p-4 glass-card border border-[var(--gold)]">
              <h4 className="text-gold font-bold mb-2">Batch Complete!</h4>
              <div className="flex-between mt-2">
                <span>Total Processed: <strong className="mono text-lg">{results.total}</strong></span>
                <span className="text-danger">Flagged: <strong className="mono text-lg">{results.flagged}</strong></span>
              </div>
            </div>
          )}
        </div>

        {/* Single Transaction Section */}
        <div className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity color="var(--navy-light)" />
            <h3>Live Simulator</h3>
          </div>
          
          <form onSubmit={handleSingleScore} className="single-score-form">
            <div className="input-group">
              <label>Amount (€)</label>
              <input 
                type="number" 
                step="0.01" 
                className="input-field mono" 
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                required
              />
            </div>
            <div className="input-group">
              <label>Time (seconds)</label>
              <input 
                type="number" 
                className="input-field mono" 
                value={time}
                onChange={(e) => setTime(e.target.value)}
                required
              />
            </div>
            <p className="text-xs text-muted mt-2">
              (Demo behavior: Amounts &gt; €1000 will simulate fraud-like PCA features to trigger the J48 rules).
            </p>
            <button 
              type="submit" 
              className="btn btn-outline w-full mt-4"
              disabled={isUploading}
            >
              Score Transaction
            </button>
          </form>

          {singleResult && (
            <div className={`single-result-card mt-6 animate-fade-in ${singleResult.decision === 'block' ? 'danger' : singleResult.decision === 'flag' ? 'warning' : 'success'}`}>
              <div className="flex-between mb-2">
                <span className="font-bold uppercase tracking-wider text-sm">
                  {singleResult.decision}
                </span>
                <span className="mono text-lg font-bold">
                  {(singleResult.fraud_probability * 100).toFixed(1)}% Risk ({singleResult.risk_level.toUpperCase()})
                </span>
              </div>
              
              <div className="result-details">
                <div className="detail-row mb-3">
                  <span className="text-muted text-xs">
                    <strong>Why {singleResult.risk_level.toUpperCase()}?</strong> 
                    {singleResult.logreg_probability < 0.40 
                      ? ' LogReg probability is safely below the 40% grey-zone threshold.'
                      : singleResult.logreg_probability > 0.75
                      ? ' LogReg probability strictly exceeds the 75% upper grey-zone threshold.'
                      : singleResult.j48_prediction === 1
                      ? ' Probability is in the grey zone (40-75%), and J48 explainable rules FLAGGED it as fraud.'
                      : ' Probability is in the grey zone, but J48 rules did NOT flag it as fraud.'}
                  </span>
                </div>
                
                <div className="detail-row">
                  <span className="text-muted">LogReg Probability:</span>
                  <span className="mono">{(singleResult.logreg_probability * 100).toFixed(2)}%</span>
                </div>
                {singleResult.j48_rule_path && (
                  <div className="detail-row flex-col items-start mt-2 border-t border-light pt-2">
                    <span className="text-muted text-xs uppercase mb-1">J48 Explainability Path:</span>
                    <span className="mono text-xs">{singleResult.j48_rule_path}</span>
                  </div>
                )}
                <div className="detail-row mt-2 text-xs">
                  <span className="text-muted">Estimated FN Cost:</span>
                  <span className="mono text-danger">€{singleResult.cost_estimate_fn}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Quick inline icon component to avoid adding more imports if not needed
const Activity = ({color}) => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>;
