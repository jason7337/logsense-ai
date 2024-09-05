"""Interactive dashboard for LogSense AI."""

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import json


class Dashboard:
    """Generate interactive visualizations for log anomaly analysis."""

    @staticmethod
    def create_anomaly_timeline(anomalies: List[Dict[str, Any]]) -> str:
        """Create timeline visualization of anomalies."""
        if not anomalies:
            return Dashboard._empty_chart("No anomalies to display")

        df = pd.DataFrame(anomalies)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()

        # Group by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            severity_data = df[df['severity'] == severity] if 'severity' in df.columns else pd.DataFrame()
            if not severity_data.empty:
                fig.add_trace(go.Scatter(
                    x=severity_data['timestamp'],
                    y=[severity] * len(severity_data),
                    mode='markers',
                    name=severity,
                    marker=dict(
                        size=10,
                        color=Dashboard._get_severity_color(severity)
                    ),
                    text=severity_data.get('type', ''),
                    hovertemplate='%{text}<br>Time: %{x}<extra></extra>'
                ))

        fig.update_layout(
            title="Anomaly Timeline",
            xaxis_title="Time",
            yaxis_title="Severity",
            height=400,
            showlegend=True,
            template="plotly_dark"
        )

        return fig.to_json()

    @staticmethod
    def create_anomaly_distribution(anomalies: List[Dict[str, Any]]) -> str:
        """Create distribution chart of anomaly types."""
        if not anomalies:
            return Dashboard._empty_chart("No anomalies to display")

        # Count anomaly types
        type_counts = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', 'unknown')
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1

        fig = go.Figure(data=[
            go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color='indianred'
            )
        ])

        fig.update_layout(
            title="Anomaly Type Distribution",
            xaxis_title="Anomaly Type",
            yaxis_title="Count",
            height=400,
            template="plotly_dark"
        )

        return fig.to_json()

    @staticmethod
    def create_severity_pie_chart(anomalies: List[Dict[str, Any]]) -> str:
        """Create pie chart of anomaly severities."""
        if not anomalies:
            return Dashboard._empty_chart("No anomalies to display")

        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        fig = go.Figure(data=[go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            hole=0.3,
            marker=dict(colors=[
                Dashboard._get_severity_color(sev) for sev in severity_counts.keys()
            ])
        )])

        fig.update_layout(
            title="Anomaly Severity Distribution",
            height=400,
            template="plotly_dark"
        )

        return fig.to_json()

    @staticmethod
    def create_hourly_heatmap(logs: List[Dict[str, Any]]) -> str:
        """Create heatmap of log activity by hour and day."""
        if not logs:
            return Dashboard._empty_chart("No logs to display")

        # Create DataFrame
        df = pd.DataFrame(logs)
        if 'timestamp' not in df.columns:
            return Dashboard._empty_chart("No timestamp data available")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()

        # Create pivot table
        pivot = df.pivot_table(
            values='line_number',
            index='day',
            columns='hour',
            aggfunc='count',
            fill_value=0
        )

        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in days_order if d in pivot.index])

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlBu_r',
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Count: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title="Log Activity Heatmap",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400,
            template="plotly_dark"
        )

        return fig.to_json()

    @staticmethod
    def create_metrics_dashboard(metrics: Dict[str, Any]) -> str:
        """Create comprehensive metrics dashboard."""
        fig = go.Figure()

        # Add gauge for detection accuracy
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('detection_accuracy', 0),
            domain={'x': [0, 0.5], 'y': [0.5, 1]},
            title={'text': "Detection Accuracy"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 90}}
        ))

        # Add number for total logs processed
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics.get('logs_processed', 0),
            domain={'x': [0.5, 1], 'y': [0.5, 1]},
            title={'text': "Logs Processed"},
            delta={'reference': metrics.get('logs_processed_yesterday', 0)}
        ))

        # Add gauge for system health
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('system_health', 100),
            domain={'x': [0, 0.5], 'y': [0, 0.5]},
            title={'text': "System Health"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "green"},
                   'bgcolor': "white",
                   'borderwidth': 2,
                   'bordercolor': "gray"}
        ))

        # Add number for anomalies detected
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics.get('anomalies_detected', 0),
            domain={'x': [0.5, 1], 'y': [0, 0.5]},
            title={'text': "Anomalies Detected"},
            delta={'reference': metrics.get('anomalies_yesterday', 0),
                   'increasing': {'color': "red"},
                   'decreasing': {'color': "green"}}
        ))

        fig.update_layout(
            height=500,
            template="plotly_dark",
            title="System Metrics Dashboard"
        )

        return fig.to_json()

    @staticmethod
    def _get_severity_color(severity: str) -> str:
        """Get color for severity level."""
        colors = {
            'CRITICAL': '#dc2626',
            'HIGH': '#ea580c',
            'MEDIUM': '#ca8a04',
            'LOW': '#16a34a',
            'UNKNOWN': '#737373'
        }
        return colors.get(severity, '#808080')

    @staticmethod
    def _empty_chart(message: str) -> str:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            template="plotly_dark",
            height=400
        )
        return fig.to_json()


def generate_html_report(analysis_results: Dict[str, Any]) -> str:
    """Generate HTML report with embedded visualizations."""
    dashboard = Dashboard()

    anomalies = analysis_results.get('details', [])

    timeline_chart = dashboard.create_anomaly_timeline(anomalies)
    distribution_chart = dashboard.create_anomaly_distribution(anomalies)
    severity_chart = dashboard.create_severity_pie_chart(anomalies)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LogSense AI Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #0284c7 0%, #9333ea 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            h1 {{ color: #333; }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric {{
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #0284c7;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #0284c7;
            }}
            .metric-label {{
                color: #666;
                margin-top: 5px;
            }}
            .chart {{
                margin: 30px 0;
            }}
            .suggestions {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            .patterns {{
                background: #d1ecf1;
                border: 1px solid #0c5460;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LogSense AI Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="summary">
                <div class="metric">
                    <div class="metric-value">{analysis_results.get('anomaly_count', 0)}</div>
                    <div class="metric-label">Anomalies Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_results.get('anomaly_percentage', 0)}%</div>
                    <div class="metric-label">Anomaly Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(analysis_results.get('patterns', []))}</div>
                    <div class="metric-label">Patterns Found</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_results.get('total_logs', 0)}</div>
                    <div class="metric-label">Total Logs Analyzed</div>
                </div>
            </div>

            <div class="suggestions">
                <h3>Recommendations</h3>
                <ul>
                    {''.join(f"<li>{s}</li>" for s in analysis_results.get('suggestions', []))}
                </ul>
            </div>

            <div class="patterns">
                <h3>Top Patterns</h3>
                <ul>
                    {''.join(f"<li><strong>{p.get('type', 'Unknown')}</strong>: {p.get('count', 0)} occurrences (Severity: {p.get('severity', 'Unknown')})</li>" for p in analysis_results.get('patterns', [])[:5])}
                </ul>
            </div>

            <div class="chart" id="timeline"></div>
            <div class="chart" id="distribution"></div>
            <div class="chart" id="severity"></div>

            <script>
                Plotly.newPlot('timeline', {timeline_chart});
                Plotly.newPlot('distribution', {distribution_chart});
                Plotly.newPlot('severity', {severity_chart});
            </script>
        </div>
    </body>
    </html>
    """

    return html_template