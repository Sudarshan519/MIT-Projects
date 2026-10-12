import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, ShieldCheck, Activity, TrendingDown,
  AlertTriangle
} from 'lucide-react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend
} from 'recharts';
import './DashboardPage.css';

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('/dashboard/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch dashboard stats', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !stats) {
    return <div className="flex-center" style={{ height: '100%' }}>Loading dashboard...</div>;
  }

  // Cost comparison chart data
  const modelsData = stats.model_comparison;
  const costChartData = [
    {
      name: 'J48 (Standard)',
      'False Positives (Friction)': modelsData.j48_standard?.fp_total_cost || 0,
      'False Negatives (Lost Money)': modelsData.j48_standard?.fn_total_cost || 0,
    },
    {
      name: 'LogReg (Tuned)',
      'False Positives (Friction)': modelsData.logreg_threshold_tuned?.fp_total_cost || 0,
      'False Negatives (Lost Money)': modelsData.logreg_threshold_tuned?.fn_total_cost || 0,
    }
  ];

  // Distribution chart data
  const statusData = [
    { name: 'Confirmed Fraud', value: stats.confirmed_frauds, color: 'var(--status-danger)' },
    { name: 'Released (Genuine)', value: stats.released, color: 'var(--status-success)' },
    { name: 'Reviewing', value: stats.active_alerts, color: 'var(--status-warning)' },
  ];

  return (
    <div className="dashboard-container animate-fade-in">
      <header className="page-header">
        <div>
          <h2>Executive Overview</h2>
          <p className="text-muted">Real-time fraud detection metrics and cost analysis.</p>
        </div>
      </header>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card glass-card delay-100">
          <div className="kpi-icon-wrapper warning">
            <AlertTriangle size={24} />
          </div>
          <div className="kpi-content">
            <p className="kpi-label">Active Alerts</p>
            <h3 className="kpi-value">{stats.active_alerts}</h3>
          </div>
        </div>

        <div className="kpi-card glass-card delay-200">
          <div className="kpi-icon-wrapper success">
            <TrendingDown size={24} />
          </div>
          <div className="kpi-content">
            <p className="kpi-label">Loss Avoided</p>
            <h3 className="kpi-value">€{stats.estimated_loss_avoided.toLocaleString()}</h3>
          </div>
        </div>

        <div className="kpi-card glass-card delay-300">
          <div className="kpi-icon-wrapper danger">
            <Activity size={24} />
          </div>
          <div className="kpi-content">
            <p className="kpi-label">False Positive Rate (FPR)</p>
            <h3 className="kpi-value">{stats.false_positive_rate.toFixed(3)}%</h3>
          </div>
        </div>

        <div className="kpi-card glass-card delay-300">
          <div className="kpi-icon-wrapper neutral">
            <ShieldCheck size={24} />
          </div>
          <div className="kpi-content">
            <p className="kpi-label">Total Scored</p>
            <h3 className="kpi-value">{stats.total_scored.toLocaleString()}</h3>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        {/* Cost Comparison Chart */}
        <div className="chart-card glass-panel">
          <h3 className="chart-title">Model Cost Evaluation (€250 FN / €15 FP)</h3>
          <p className="chart-subtitle">Cost-sensitive analysis showing total operating cost</p>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={costChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-muted)" />
                <YAxis stroke="var(--text-muted)" tickFormatter={(val) => `€${val/1000}k`} />
                <Tooltip 
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
                />
                <Legend />
                <Bar dataKey="False Positives (Friction)" stackId="a" fill="var(--status-warning)" />
                <Bar dataKey="False Negatives (Lost Money)" stackId="a" fill="var(--status-danger)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Status Distribution */}
        <div className="chart-card glass-panel">
          <h3 className="chart-title">Alert Resolution Status</h3>
          <div className="chart-wrapper flex-center">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
                />
                <Legend verticalAlign="bottom" height={36} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
