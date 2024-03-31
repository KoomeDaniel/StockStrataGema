function transition(clicked_id) {
    let main_links = ['dashboard', 'watchlist', 'marketnews'];
    if (main_links.includes(clicked_id)) {
        document.body.classList.add("w3-animate-opacity");
    }
}

$(document).ready(function() {

    // Show the body on page load 
    $('body').show(0);
    
    // Bootstrap Tooltip
    $('[data-toggle="tooltip"]').tooltip();

    // Toggle detween financials divs
    $('button').on('click', function() {
        let btns = ["balance-btn", "income-btn", "cashflow-btn"];
        let divs = ["balance-sheet", "income-statement", "cashflow-statement"];

        if (btns.includes(this.id)) {
            for (let i = 0; i < btns.length; i++) {
                if (btns[i] == this.id) {
                    $("#" + divs[i]).css({ "display": "block" });
                    $("#" + btns[i]).removeClass("btn-outline-info");
                    $("#" + btns[i]).addClass("btn-info");
                } else {
                    $("#" + divs[i]).css({ "display": "none" });
                    $("#" + btns[i]).removeClass("btn-info");
                    $("#" + btns[i]).addClass("btn-outline-info");
                }
            }
        }
    });

    // Toggle detween news divs
    $('button').on('click', function() {
        let btns = ["google-btn", "marketwatch-btn", "businessinsider-btn", "financialpost-btn"];
        let divs = ["google-news", "marketwatch-news", "businessinsider-news", "financialpost-news"];

        if (btns.includes(this.id)) {
            for (let i = 0; i < btns.length; i++) {
                if (btns[i] == this.id) {
                    $("#" + divs[i]).css({ "display": "block" });
                    $("#" + btns[i]).removeClass("btn-outline-info");
                    $("#" + btns[i]).addClass("btn-info");
                } else {
                    $("#" + divs[i]).css({ "display": "none" });
                    $("#" + btns[i]).removeClass("btn-info");
                    $("#" + btns[i]).addClass("btn-outline-info");
                }
            }
        }
    });

    
    // Populate Search Bar Autocomplete 
    $(function (){
        var companies = search_data;
        $( "#search-bar" ).autocomplete({
            minLength: 2,   
            source: companies,
            autoFocus:true
        }).autocomplete( "instance" )._renderItem = function( ul, item ) {
            return $( "<li><a href='#'>" + item.label + "</li>").appendTo( ul );
          };
    });

    // Submit ticker for search bar on enter 
    $('#search-bar').keypress(function(e) {
        var key = e.which;
        if (key == 13) // the enter key code
        {
            $('button[id=search-bar-button]').click();
            return false;
        }
    });

    // Submit ticker for search bar on button click 
    $('#search-bar-button').on('click', function(e) {
        let ticker_search_enter = $('#search-bar').val();
        if(ticker_search_enter){
            return;
        }
    });
});