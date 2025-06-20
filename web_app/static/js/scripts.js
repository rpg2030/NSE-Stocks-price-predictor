// Store the original list of stocks from the dropdown
const stockItems = Array.from(document.querySelectorAll('.dropdown-item')).map(item => item.dataset.value);
let selectedStock = stockItems[0] || ''; // Default to first stock if available

// Get DOM elements
const stockInput = document.getElementById('stockInput');
const stockDropdown = document.getElementById('stockDropdown');
const stockForm = document.getElementById('stockForm');
const daysInput = document.getElementById('daysInput');
const chartContainer = document.getElementById('chartContainer');
const stockChartCanvas = document.getElementById('stockChart');
const metricsContainer = document.getElementById('metrics');
let stockChart = null; // Chart.js instance

// Show/hide dropdown
function toggleDropdown(show) {
    stockDropdown.classList.toggle('show', show);
}

// Filter stocks based on input
function filterStocks() {
    const query = stockInput.value.toLowerCase();
    stockDropdown.innerHTML = ''; // Clear current dropdown items
    const filteredStocks = stockItems.filter(stock => stock.toLowerCase().startsWith(query));
    
    filteredStocks.forEach(stock => {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.dataset.value = stock;
        item.textContent = stock;
        item.addEventListener('click', () => {
            stockInput.value = stock;
            selectedStock = stock;
            toggleDropdown(false);
        });
        stockDropdown.appendChild(item);
    });

    toggleDropdown(filteredStocks.length > 0 && query.length > 0);
}

// Event listeners for input
stockInput.addEventListener('input', filterStocks);
stockInput.addEventListener('focus', () => filterStocks());
stockInput.addEventListener('click', () => filterStocks());
document.addEventListener('click', (e) => {
    if (!stockInput.contains(e.target) && !stockDropdown.contains(e.target)) {
        toggleDropdown(false);
    }
});

// Form submission
stockForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedStock) {
        alert('Please select a stock');
        return;
    }
    const days = parseInt(daysInput.value);
    if (isNaN(days) || days <= 0) {
        alert('Please enter a valid number of days');
        return;
    }

    try {
        // Fetch original data
        const originalResponse = await fetch(`/original/${selectedStock}?days=${days}`);
        if (!originalResponse.ok) {
            throw new Error((await originalResponse.json()).error);
        }
        const originalData = await originalResponse.json();

        // Fetch predicted data
        const predictResponse = await fetch(`/predict/${selectedStock}/${days}`);
        if (!predictResponse.ok) {
            throw new Error((await predictResponse.json()).error);
        }
        const predictedData = await predictResponse.json();

        // Fetch evaluation metrics
        const evaluateResponse = await fetch(`/evaluate/${selectedStock}`);
        if (!evaluateResponse.ok) {
            throw new Error((await evaluateResponse.json()).error);
        }
        const metrics = await evaluateResponse.json();

        // Update chart
        chartContainer.style.display = 'block';
        updateChart(originalData, predictedData);

        // Update metrics
        updateMetrics(metrics);
    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error('Error:', error);
    }
});

// Update Chart.js chart
function updateChart(originalData, predictedData) {
    const labels = [...originalData.dates, ...predictedData.dates];
    const data = [...originalData.close_prices, ...predictedData.predictions];

    if (stockChart) {
        stockChart.destroy(); // Destroy existing chart
    }

    stockChart = new Chart(stockChartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Prices',
                    data: originalData.close_prices,
                    borderColor: '#007bff',
                    fill: false
                },
                {
                    label: 'Predicted Prices',
                    data: [...new Array(originalData.close_prices.length).fill(null), ...predictedData.predictions],
                    borderColor: '#ff4500',
                    fill: false,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                }
            }
        }
    });
}

// Update metrics display
function updateMetrics(metrics) {
    // Check if metrics and its properties exist, provide fallback values
    const mae = metrics && typeof metrics.mae === 'number' ? metrics.mae.toFixed(2) : 'N/A';
    // const mse = metrics && typeof metrics.mse === 'number' ? metrics.mse.toFixed(2) : 'N/A';
    const rmse = metrics && typeof metrics.rmse === 'number' ? metrics.rmse.toFixed(2) : 'N/A';
    const mape = metrics && typeof metrics.mape === 'number' ? metrics.mape.toFixed(2) : 'N/A';
    const warning = metrics && metrics.warning ? metrics.warning : '';

    metricsContainer.innerHTML = `
        <h3>Model Performance</h3>
        <p>MAE: ${mae}</p>
        <p>MAPE: ${mape}%</p>
        <p>RMSE: ${rmse}</p>
        <p style="color: red;">${warning}</p>
    `;
}