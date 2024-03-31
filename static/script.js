var menubar = document.querySelector('#menu-bar')
var mynav = document.querySelector('.navbar')


menubar.onclick = () =>{
    menubar.classList.toggle('fa-times')
    mynav.classList.toggle('active')
}

const navbar = document.querySelector('.header')

window.onscroll = () =>{
    if(window.scrollY > 300){
        navbar.classList.add('active')
    }
    else{
        navbar.classList.remove('active')
    }
}
window.onload = displayGainers; // Display gainers by default

function displayGainers() {
    var top_gainers = JSON.parse(document.getElementById('top-gainers-data').textContent);
    var table = createTable(top_gainers, 'gainers');
    var tableContainer = document.getElementById('table-container');
    tableContainer.innerHTML = ''; // Clear the table container
    tableContainer.innerHTML = table; // Insert the new table
    tableContainer.style.display = 'block'; // Display the table container
}

function displayLosers() {
    var top_losers = JSON.parse(document.getElementById('top-losers-data').textContent);
    var table = createTable(top_losers, 'losers');
    var tableContainer = document.getElementById('table-container');
    tableContainer.innerHTML = ''; // Clear the table container
    tableContainer.innerHTML = table; // Insert the new table
    tableContainer.style.display = 'block'; // Display the table container
}

function createTable(data, type) {
    var table = '<table class="' + type + '">';
    if (type === 'gainers') {
        table += '<tr><th>Company</th><th>High</th><th>Low</th><th>Change</th><th>Gain in %</th><th>Close Price</th></tr>';
        for (var i = 0; i < data.length; i++) {
            table += '<tr><td>' + data[i].company + '</td><td>' + data[i].high + '</td><td>' + data[i].low + '</td><td>' + data[i].change + '</td><td>' + data[i].gain_in_per + '</td><td>' + data[i].close_in_per + '</td></tr>';
        }
    } else if (type === 'losers') {
        table += '<tr><th>Company</th><th>High</th><th>Low</th><th>Change</th><th>Loss in %</th><th>Close Price</th></tr>';
        for (var i = 0; i < data.length; i++) {
            table += '<tr><td>' + data[i].company + '</td><td>' + data[i].High + '</td><td>' + data[i].Low + '</td><td>' + data[i].Change + '</td><td>' + data[i].Loss_in_per + '</td><td>' + data[i].close_price + '</td></tr>';
        }
    }
    table += '</table>';
    return table;
}
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
document.getElementById('user-bar').addEventListener('click', function() {
    document.getElementById('popup').style.display = 'block';
    document.getElementById('user-bar').style.display = 'none';
    document.getElementById('cancel-bar').style.display = 'block';
});

document.getElementById('cancel-bar').addEventListener('click', function() {
    document.getElementById('popup').style.display = 'none';
    document.getElementById('user-bar').style.display = 'block';
    document.getElementById('cancel-bar').style.display = 'none';
});
var navLinks = document.querySelectorAll('.navbar a');

// Function to remove 'active' class from all navigation links
function removeActiveClass() {
    navLinks.forEach(function(link) {
        link.classList.remove('active');
    });
}

// Function to add 'active' class to a navigation link
function addActiveClass(index) {
    navLinks[index].classList.add('active');
}

// Add 'click' event listeners to all navigation links
navLinks.forEach(function(link, index) {
    link.addEventListener('click', function() {
        removeActiveClass();
        addActiveClass(index);
    });
});
// Get all the sections
var sections = document.querySelectorAll('section');

// Get all the navigation links
var navLinks = document.querySelectorAll('.navbar a');

// Function to remove 'active' class from all navigation links
function removeActiveClass() {
    navLinks.forEach(function(link) {
        link.classList.remove('active');
    });
}

// Function to add 'active' class to a navigation link
function addActiveClass(index) {
    navLinks[index].classList.add('active');
}

// Create an intersection observer
var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        if (entry.isIntersecting) {
            var index = Array.prototype.indexOf.call(sections, entry.target);
            removeActiveClass();
            addActiveClass(index);
        }
    });
}, { threshold: 0.5 }); // Adjust the threshold value to control when the 'active' class is added

// Observe each section
sections.forEach(function(section) {
    observer.observe(section);
});