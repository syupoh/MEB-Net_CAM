 # ! / u s r / b i n   b a s h 
 # ! / b i n / b a s h 
 
 i f   [   - z   " $ 1 "   ] 
     t h e n 
         g p u = " 0 " 
     e l s e 
         g p u = " $ { 1 } " 
 f i 
 
 i f   [   $ { g p u }   - e q   " 0 "   ] 
     t h e n 
 
 # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # 
         p a s s 
     e l i f   [   $ { g p u }   - e q   " 1 "   ] 
 
         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s   . / g r a d c a m / i m g s / d u k e m t m c ' 
 #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s ' 
 
         f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
         d o 
             p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                   - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
         d o n e 
 
     t h e n 
 
 # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # 
         p a s s 
     e l i f     [   $ { g p u }   - e q   " 2 "   ] 
     t h e n 
         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s   . / g r a d c a m / i m g s / d u k e m t m c ' 
 #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s ' 
 
         f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
         d o 
             p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                   - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
         d o n e 
 
 # # # # # # # # # # # # # # # # # 
         p a s s 
 f i 
 
 
 #         s e a r c h _ d i r = ' . / g r a d c a m / d u k e m t m c   . / g r a d c a m / m a r k e t ' 
 #         m o d e l _ d i r _ s e t = ' . / l o g s / _ 0 5 0 5 _ t y p e F ' 
 # 
 #         f o r   m o d e l _ d i r   i n   $ { m o d e l _ d i r _ s e t } 
 #         d o 
 #             f o r   m o d e l _ n a m e   i n   ` l s   $ { m o d e l _ d i r } ` 
 #             d o 
 #                 f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
 #                 d o 
 #                     f o r   i m a g e   i n   ` l s   $ { i m a g e _ d i r } ` 
 #                     d o 
 #                         p y t h o n   g r a d c a m _ d . p y   - - g p u = $ { g p u }   \ 
 #                           - - m o d e l = $ { m o d e l _ d i r } / $ { m o d e l _ n a m e } / c h e c k p o i n t _ 7 9 . p t h . t a r   - - i m a g e - p a t h = $ { i m a g e _ d i r } / $ { i m a g e } 
 #                     d o n e 
 #                 d o n e 
 #             d o n e 
 #         d o n e 
 
 
 p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / r e s n e t _ m a r k e t . p t h . t a r   - - g p u   $ { g p u } 
 # p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / d e n s e n e t _ m a r k e t . p t h . t a r 
 # p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / i n c e p t i o n v 3 _ m a r k e t . p t h . t a r 
 
 # i m g _ s e t = ' p e r s o n 1 . j p g   p e r s o n 2 . j p g   p e r s o n 3 . j p g   p e r s o n 4 . j p g   p e r s o n 5 . j p g   p e r s o n 6 . j p g   p e r s o n 7 . j p g   p e r s o n 8 . j p g ' 
 # 
 # f o r   i m g   i n   $ { i m g _ s e t } 
 # d o 
 #     p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / $ { i m g }   - - m o d e l   . / g r a d c a m / r e s n e t _ m a r k e t . p t h . t a r 
 # d o n e 
 # 
