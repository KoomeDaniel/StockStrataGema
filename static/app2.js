window.onload = function() {
  if (!window.location.hash || window.location.hash !== '#home-section') {
    window.location.hash = 'home-section';
  }
   // JavaScript to handle image upload and update user-icon image
   document.getElementById('image-upload-form').addEventListener('submit', function(event) {
    // Prevent the default form submission
    event.preventDefault();

    // Get the form data
    var formData = new FormData(document.getElementById('image-upload-form'));

    // Make an asynchronous POST request to the server
    fetch('/upload-image', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.text())
    .then(data => {
        // Display the server response (you can modify this part as needed)
        alert(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
  document.querySelector('.icons a img[alt="User"]').addEventListener('click', function(event) {
    event.preventDefault();
    document.getElementById('user-popup').style.display = 'block';
});

  // Add event listener for the close button
  document.getElementById('close-btn').addEventListener('click', function() {
      document.getElementById('user-popup').style.display = 'none';
  });

  // Add event listener for the search icon
  document.querySelector('.icons a img[alt="Search"]').addEventListener('click', function(event) {
      event.preventDefault();
      document.getElementById('search-box').style.display = 'block';
  });

  // Add event listener for the search input
  document.getElementById('search-input').addEventListener('input', function(e) {
      console.log('Search query:', e.target.value);
      // Add your search logic here
  });

  // Add event listener for the search button
  document.getElementById('search-btn').addEventListener('click', function() {
      console.log('Search button clicked');
      // Add your search logic here
  });

  // Add event listener for the close search button
  document.getElementById('close-search-btn').addEventListener('click', function() {
      document.getElementById('search-box').style.display = 'none';
  });

  // Add event listener for the menu icon
  document.querySelector('.icons a img[alt="Menu"]').addEventListener('click', function(event) {
      event.preventDefault();
      document.getElementById('sidebar').style.display = 'block';
  });

  // Add event listener for the close sidebar button
  document.getElementById('close-sidebar-btn').addEventListener('click', function() {
      document.getElementById('sidebar').style.display = 'none';
  });

  let headerLinks = document.querySelectorAll('.nav-links a');
  let sidebarLinks = document.querySelectorAll('#sidebar a');

  // Function to remove active class from all elements
  function removeActiveClass(elements) {
    elements.forEach(element => {
      element.classList.remove('active');
    });
  }

  // Function to add active class to clicked element
  function addActiveClass(element) {
    removeActiveClass(headerLinks);
    removeActiveClass(sidebarLinks);
    element.classList.add('active');
  }

  // Add click event listener to all header links
  headerLinks.forEach(link => {
    link.addEventListener('click', function() {
      addActiveClass(this);
    });
  });

  // Set "Home" link as active by default
  addActiveClass(document.querySelector('.nav-links a[href="#home-section"]'));

 
  let userPopupClickCount = 0;
  let searchBoxClickCount = 0;
  let sidebarClickCount = 0;
  document.querySelector('.icons a img[alt="User"]').addEventListener('click', function() {
    userPopupClickCount++;

    // Toggle user popup visibility on double click
    if (userPopupClickCount === 2) {
      document.getElementById('user-popup').style.display = 'none';
      userPopupClickCount = 0; // Reset click count
    }
  });

  document.querySelector('.icons a img[alt="Search"]').addEventListener('click', function() {
    searchBoxClickCount++;

    // Toggle search box visibility on double click
    if (searchBoxClickCount === 2) {
      document.getElementById('search-box').style.display = 'none';
      searchBoxClickCount = 0; // Reset click count
    }
  });

  document.querySelector('.icons a img[alt="Menu"]').addEventListener('click', function() {
    sidebarClickCount++;

    // Toggle sidebar visibility on double click
    if (sidebarClickCount === 2) {
      document.getElementById('sidebar').style.display = 'none';
      sidebarClickCount = 0; // Reset click count
    }
  });      
  var form = document.querySelector('.symbol');
  var input = document.querySelector('input[name="nm"]');
  var select = document.querySelector('select[name="timeframe"]');
  var submit = document.querySelector('input[type="submit"]');

  function checkFields() {
      if (input.value && select.value) {
          submit.disabled = false;
      } else {
          submit.disabled = true;
      }
  }

  input.addEventListener('input', function(e) {
      e.target.value = e.target.value.toUpperCase();
      checkFields();
  });

  select.addEventListener('change', checkFields);  

  show('gainer');
};

function show(type) {
  // Get all table rows
  var rows = document.getElementById('stock-table').getElementsByTagName('tr');

  // Loop through the rows and hide or show them based on the type
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].classList.contains(type)) {
      rows[i].style.display = '';  // Show row
    } else {
      rows[i].style.display = 'none';  // Hide row
    }
  }  
}


const apiKey = '6ET31V3K79OLGGJO';
const tableBody = document.querySelector('#stock-table tbody');

const fetchData = async () => {
  const response = await fetch(
    `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT,AAPL,GOOGL,AMZN,HDFCBANK,TATAMOTORS,ICICIBANK,JPM,FB,TWTR,ZEEL,NTPC,COALINDIA,YESBANK&apikey=${apiKey}`
  );
  const data = await response.json();
  return data;
};

const populateTable = (data) => {
  tableBody.innerHTML = '';
  const stocks = Object.keys(data['Time Series (Daily)']);
  
  // Create an array to hold the stocks and their changes
  let stockChanges = [];
  
  stocks.forEach((stock) => {
    const close = parseFloat(data['Time Series (Daily)'][stock]['4. close']);
    const volume = parseFloat(data['Time Series (Daily)'][stock]['5. volume']);
    const volumeChange = volume - close;
    const percentChange = (volumeChange / close) * 100;
    
    // Add the stock and its change to the array
    stockChanges.push({stock, volumeChange, percentChange});
  });
  
  // Sort the array by the percent change
  stockChanges.sort((a, b) => b.percentChange - a.percentChange);
  
  // Now populate the table with the sorted data
  stockChanges.forEach(({stock, volumeChange, percentChange}) => {
    const row = document.createElement('tr');
    
    const nameCell = document.createElement('td');
    nameCell.textContent = stock;
    row.appendChild(nameCell);
    
    const valueCell = document.createElement('td');
    valueCell.textContent = close.toFixed(2);
    row.appendChild(valueCell);
    
    const changeCell = document.createElement('td');
    changeCell.textContent = volumeChange.toFixed(2);
    row.appendChild(changeCell);
    
    const percentChangeCell = document.createElement('td');
    percentChangeCell.textContent = `${percentChange.toFixed(2)}%`;
    row.appendChild(percentChangeCell);
    
    tableBody.appendChild(row);
  });
};

        