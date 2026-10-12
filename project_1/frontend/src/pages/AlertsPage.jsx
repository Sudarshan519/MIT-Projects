import { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, CheckCircle, ArrowRight, XCircle, Search } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './AlertsPage.css';

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const navigate = useNavigate();

  useEffect(() => {
    fetchAlerts();
  }, [filter]);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      // For demo, we are just fetching the recent transactions
      // In a real app, we'd pass pagination and filters
      const response = await axios.get('/transactions', {
        params: { status: filter === 'all' ? null : filter, per_page: 50 }
      });
      setAlerts(response.data.items);
    } catch (error) {
      console.error('Failed to fetch alerts', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusUpdate = async (id, targetStatus) => {
    try {
      await axios.patch(`/transactions/${id}/status`, {
        status: targetStatus,
        notes: `Analyst manually transitioned to ${targetStatus}`
      });
      // Refresh list
      fetchAlerts();
    } catch (error) {
      console.error('Failed to update status', error);
      alert('Failed to update status.');
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'new': return <span className="badge badge-danger">New Alert</span>;
      case 'reviewing': return <span className="badge badge-warning">Reviewing</span>;
      case 'confirmed_fraud': return <span className="badge badge-neutral">Confirmed</span>;
      case 'released': return <span className="badge badge-success">Released</span>;
      default: return <span className="badge badge-neutral">{status}</span>;
    }
  };

  const getRiskBadge = (risk) => {
    switch (risk) {
      case 'CRITICAL': return <span className="badge badge-danger">Critical</span>;
      case 'HIGH': return <span className="badge badge-warning">High</span>;
      case 'MEDIUM': return <span className="badge badge-neutral">Medium</span>;
      case 'LOW': return <span className="badge badge-success">Low</span>;
      default: return <span className="badge badge-neutral">{risk}</span>;
    }
  };

  return (
    <div className="alerts-container animate-fade-in">
      <header className="page-header flex-between">
        <div>
          <h2>Alert Queue</h2>
          <p className="text-muted">Review and resolve flagged transactions.</p>
        </div>
        
        <div className="filter-group">
          <select 
            className="input-field" 
            value={filter} 
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '180px' }}
          >
            <option value="all">All Alerts</option>
            <option value="new">New Alerts</option>
            <option value="reviewing">Under Review</option>
            <option value="confirmed_fraud">Confirmed Fraud</option>
          </select>
        </div>
      </header>

      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div className="table-responsive">
          <table className="alerts-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Time (s)</th>
                <th>Amount</th>
                <th>Risk Level</th>
                <th>LogReg Score</th>
                <th>J48 Rule Trigger</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="8" className="text-center py-4">Loading alerts...</td></tr>
              ) : alerts.length === 0 ? (
                <tr><td colSpan="8" className="text-center py-4 text-muted">No alerts found in this category.</td></tr>
              ) : (
                alerts.map((alert) => (
                  <tr key={alert.id} className="table-row">
                    <td className="mono">#{alert.id}</td>
                    <td className="mono">{alert.time?.toFixed(0)}</td>
                    <td className="mono font-bold">€{alert.amount?.toFixed(2)}</td>
                    <td>{getRiskBadge(alert.risk_level?.toUpperCase())}</td>
                    <td>
                      <div className="score-bar-bg">
                        <div 
                          className="score-bar-fill" 
                          style={{ 
                            width: `${alert.logreg_probability * 100}%`,
                            background: alert.logreg_probability > 0.5 ? 'var(--status-danger)' : 'var(--status-success)'
                          }}
                        />
                      </div>
                      <span className="mono text-xs">{(alert.logreg_probability * 100).toFixed(1)}%</span>
                    </td>
                    <td className="text-xs text-muted max-w-xs truncate">
                      {alert.j48_rule_path || 'None'}
                    </td>
                    <td>{getStatusBadge(alert.status)}</td>
                    <td>
                      <div className="flex gap-2">
                        {alert.status === 'new' && (
                          <>
                            <button className="btn btn-warning btn-sm" onClick={() => handleStatusUpdate(alert.id, 'reviewing')}>
                              Review
                            </button>
                            <button className="btn btn-success btn-sm" onClick={() => handleStatusUpdate(alert.id, 'released')}>
                              Dismiss
                            </button>
                          </>
                        )}
                        {alert.status === 'reviewing' && (
                          <>
                            <button className="btn btn-danger btn-sm" onClick={() => handleStatusUpdate(alert.id, 'confirmed_fraud')}>
                              Confirm Fraud
                            </button>
                            <button className="btn btn-success btn-sm" onClick={() => handleStatusUpdate(alert.id, 'released')}>
                              False Alarm
                            </button>
                          </>
                        )}
                        {alert.status === 'confirmed_fraud' && (
                          <button className="btn btn-success btn-sm" onClick={() => handleStatusUpdate(alert.id, 'released')}>
                            Unblock Card
                          </button>
                        )}
                        {alert.status === 'released' && (
                          <span className="text-xs text-muted">Closed</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
