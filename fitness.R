

R <- seq(0.0001,0.9999,0.0001)

a1 <- 0.1
a2 <- 0.01
f1 <- R**a1
f2 <- R**a2

b1 <- 0.1
b2 <- 0.01
g1 <- (1 - exp(-R/b1)) / (1 - exp(-1/b1))
g2 <- (1 - exp(-R/b2)) / (1 - exp(-1/b2))

pdf("fita.pdf")
plot(f1~R,cex=0,ylab="F(R)",cex.lab=1.5)
lines(f1~R,lwd=4)
lines(f2~R,lwd=4)
text(0.5,0.9,expression(~alpha==0.1),cex=1.5)
text(0.2,0.95,expression(~alpha==0.01),cex=1.5)
dev.off()

pdf("fitb.pdf")
plot(g1~R,cex=0,ylab="F(R)",cex.lab=1.5)
lines(g1~R,lwd=4)
lines(g2~R,lwd=4)
text(0.4,0.9,expression(~beta==0.1),cex=1.5)
text(0.15,0.95,expression(~beta==0.01),cex=1.5)
dev.off()

pdf("fitc.pdf")
plot(log(f1)~R,cex=0,ylab="F(R)",cex.lab=1.5)
lines(log(f1)~R,lwd=4)
lines(log(f2)~R,lwd=4)
dev.off()

pdf("fitd.pdf")
plot(log(g1)~R,cex=0,ylab="F(R)",cex.lab=1.5)
lines(log(g1)~R,lwd=4)
lines(log(g2)~R,lwd=4)
dev.off()



