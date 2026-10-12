import { useState, useEffect } from 'react';
import axios from 'axios';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Settings2, TrendingDown } from 'lucide-react';

export default function CostAnalysisPage() {
  const [data, setData] = useState(null);
  const [fnCost, setFnCost] = useState(250);
  const [fpCost, setFpCost] = useState(15);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCostData();
  }, [fnCost, fpCost]);

  const fetchCostData = async () => {
    try {
      const response = await axios.get('/dashboard/cost-analysis', {
        params: { fn_cost: fnCost, fp_cost: fpCost }
      });
      setData(response.data);
    } catch (error) {
      console.error('Failed to fetch cost analysis', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !data) return <div className="flex-center" style={{height: '100%'}}>Loading...</div>;

  const modelsData = data?.models || {};
  
  // Format for the chart
  const chartData = [
    {
      name: 'J48 (Standard)',
      'False Positives (Friction)': modelsData.j48_standard?.fp_total_cost || 0,
      'False Negatives (Loss)': modelsData.j48_standard?.fn_total_cost || 0,
      total: modelsData.j48_standard?.total_cost || 0,
    },
    {
      name: 'J48 (Cost-Sens)',
      'False Positives (Friction)': modelsData.j48_cost_sensitive?.fp_total_cost || 0,
      'False Negatives (Loss)': modelsData.j48_cost_sensitive?.fn_total_cost || 0,
      total: modelsData.j48_cost_sensitive?.total_cost || 0,
    },
    {
      name: 'LogReg (Std)',
      'False Positives (Friction)': modelsData.logreg_standard?.fp_total_cost || 0,
      'False Negatives (Loss)': modelsData.logreg_standard?.fn_total_cost || 0,
      total: modelsData.logreg_standard?.total_cost || 0,
    },
    {
      name: 'LogReg (Tuned)',
      'False Positives (Friction)': modelsData.logreg_threshold_tuned?.fp_total_cost || 0,
      'False Negatives (Loss)': modelsData.logreg_threshold_tuned?.fn_total_cost || 0,
      total: modelsData.logreg_threshold_tuned?.total_cost || 0,
    }
  ];

  const winner = modelsData._winner;
  const savings = modelsData._savings;

  return (
    <div className="dashboard-container animate-fade-in">
      <header className="page-header">
        <div>
          <h2>Interactive Cost Analysis</h2>
          <p className="text-muted">Simulate different cost assumptions to evaluate model business impact.</p>
        </div>
      </header>

      <div className="charts-grid" style={{ gridTemplateColumns: '1fr 3fr' }}>
        
        {/* Sliders Panel */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid var(--border-light)', paddingBottom: '1rem' }}>
            <Settings2 size={20} color="var(--gold)" />
            <h3 style={{ fontSize: '1.125rem' }}>Cost Assumptions</h3>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div className="flex-between">
              <label style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>False Negative Cost</label>
              <span className="mono font-bold">€{fnCost}</span>
            </div>
            <input 
              type="range" 
              min="50" max="1000" step="10"
              value={fnCost} 
              onChange={(e) => setFnCost(Number(e.target.value))}
              style={{ accentColor: 'var(--status-danger)' }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cost of missed fraud (loss + chargeback)</p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div className="flex-between">
              <label style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>False Positive Cost</label>
              <span className="mono font-bold">€{fpCost}</span>
            </div>
            <input 
              type="range" 
              min="1" max="100" step="1"
              value={fpCost} 
              onChange={(e) => setFpCost(Number(e.target.value))}
              style={{ accentColor: 'var(--status-warning)' }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cost of false alarm (support + friction)</p>
          </div>

          <div className="glass-card" style={{ padding: '1rem', marginTop: 'auto', background: 'var(--status-success-bg)', borderColor: 'var(--status-success)' }}>
            <h4 style={{ fontSize: '0.875rem', color: 'var(--status-success)', marginBottom: '0.5rem' }}>Recommended Model</h4>
            <p style={{ fontSize: '1.125rem', fontWeight: 'bold' }}>
              {winner === 'j48_standard' ? 'J48 (Standard)' : 
               winner === 'j48_cost_sensitive' ? 'J48 (Cost-Sens)' :
               winner === 'logreg_standard' ? 'LogReg (Std)' : 'LogReg (Tuned)'}
            </p>
            <p style={{ fontSize: '0.75rem', marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <TrendingDown size={14} /> Saves €{savings?.toLocaleString()} vs worst
            </p>
          </div>
        </div>

        {/* Chart Panel */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '1.125rem', marginBottom: '1.5rem' }}>Total Operating Cost by Model</h3>
          <div style={{ flex: 1, width: '100%', minHeight: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-muted)" />
                <YAxis stroke="var(--text-muted)" tickFormatter={(val) => `€${val/1000}k`} />
                <Tooltip 
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
                  formatter={(value) => `€${value.toLocaleString()}`}
                />
                <Legend />
                <Bar dataKey="False Positives (Friction)" stackId="a" fill="var(--status-warning)" />
                <Bar dataKey="False Negatives (Loss)" stackId="a" fill="var(--status-danger)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
