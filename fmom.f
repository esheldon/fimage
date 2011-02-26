c vim: set filetype=fortran et ts=2 sw=2 sts=2 :

      subroutine mom_bilin(image,nx,ny,nsub,cen,cov)

      ! image must be sky subtracted

      implicit none

      integer*4 nx,ny
      real*4 image(nx,ny)
      real*8 cen(2), cov(3)

      real*8 xcen, ycen
      integer*4 nsub

      real*8 ixx,ixy,iyy

      integer*4 ix,iy,iix,iiy

      real*8 stepsize, offset
      real*8 x,y
      integer*4 x1,x2,y1,y2

      real*8 isum, xsum, ysum, xxsum, xysum, yysum
      real*8 val,val11,val12,val21,val22
      real*8 fac

      stepsize = 1./nsub
      offset = (nsub-1)*stepsize/2.


      ! first get the center without sub-pixel
      isum=0
      xsum=0
      ysum=0
      xxsum=0
      xysum=0
      yysum=0

      if (nsub .eq. 1) then
        do ix=1,nx
          do iy=1,ny
            val = image(ix,iy)
            x=float(ix)
            y=float(iy)
            isum = isum + val
            xsum = xsum + val*x
            ysum = ysum + val*y

            xxsum = xxsum + x*x*val
            xysum = xysum + x*y*val
            yysum = yysum + y*y*val

          enddo
        enddo

        xcen = xsum/isum
        ycen = ysum/isum

        ixx = xxsum/isum - xcen**2
        ixy = xysum/isum - xcen*ycen
        iyy = yysum/isum - ycen**2
        goto 8080
      endif

      ! interpolate across pixel except when at the edge
      do ix=1,nx
        do iy=1,ny


            x=ix-offset
            do iix=1,nsub
              y=iy-offset
              do iiy=1,nsub
                ! interpolate across pixel 

                ! locations for source pixels in x
                if (ix.eq.1) then
                  x1=1
                  x2=2
                else if (ix.eq.nx) then
                  x1=nx-1
                  x2=nx
                else
                  if (x.le.ix) then
                    x1=ix-1
                    x2=ix
                  else
                    x1=ix
                    x2=ix+1
                  endif
                endif

                ! locations for source pixels in y
                if (iy.eq.1) then
                  y1=1
                  y2=2
                else if (iy.eq.ny) then
                  y1=ny-1
                  y2=ny
                else
                  if (y.le.iy) then
                    y1=iy-1
                    y2=iy
                  else
                    y1=iy
                    y2=iy+1
                  endif
                endif

                val11 = image(x1,y1)
                val12 = image(x1,y2)
                val21 = image(x2,y1)
                val22 = image(x2,y2)

                fac=1./(x2-x1)/(y2-y1)
                val11 = val11*(x2-x)*(y2-y)*fac
                val12 = val12*(x2-x)*(y-y1)*fac
                val21 = val21*(x-x1)*(y2-y)*fac
                val22 = val22*(x-x1)*(y-y1)*fac

                val=val11+val12+val21+val22

                !frac{f(Q_{11})}{(x_2-x_1)(y_2-y_1)} (x_2-x)(y_2-y) \\
                !frac{f(Q_{12})}{(x_2-x_1)(y_2-y_1)} (x_2-x)(y-y_1) \\
                !frac{f(Q_{21})}{(x_2-x_1)(y_2-y_1)} (x-x_1)(y_2-y) \\
                !frac{f(Q_{22})}{(x_2-x_1)(y_2-y_1)} (x-x_1)(y-y_1)

                isum = isum + val
                xsum = xsum + x*val
                ysum = xsum + y*val

                xxsum = xxsum + x*x*val
                xysum = xysum + x*y*val
                yysum = yysum + y*y*val


                y = y+stepsize
              enddo
              x=x+stepsize
            enddo

        enddo
      enddo

      xcen = xsum/isum
      ycen = ysum/isum
      ixx = xxsum/isum - xcen**2
      ixy = xysum/isum - xcen*ycen
      iyy = yysum/isum - ycen**2

 8080 continue
      cen(1) = xcen
      cen(2) = ycen

      cov(1) = ixx
      cov(2) = ixy
      cov(3) = iyy

      return
      end

