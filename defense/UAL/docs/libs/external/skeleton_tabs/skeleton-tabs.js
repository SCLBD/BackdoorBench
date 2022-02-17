
$(function() {
    $('ul.tab-nav li .button').click(function() {
        var href = $(this).attr('data-ref');

        $('li .active.button', $(this).parent().parent()).removeClass('active');
        $(this).addClass('active');

        $('.tab-pane.active', $(href).parent()).removeClass('active');
        $(href).addClass('active');

        /*
        var toScroll = $(this).parent().parent().parent().parent();

        $('html, body').animate({
    		scrollTop: toScroll.offset().top
		}, 1000);
		*/

        return false;
    });
});
