import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioResult;
import engine.helper.GameStatus;

public class PlayLevel {
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    public static void main(String[] args) {
        MarioGame game = new MarioGame();
        // printResults(game.playGame(getLevel("../levels/original/lvl-1.txt"), 200, 0));
        try{
            int wins = 0;
            File folder = new File("../../generated_levels/analysis/not_trained");
            File[] listOfFiles = folder.listFiles();
            for (File levelFile: listOfFiles) {
                MarioResult result = game.runGame(new agents.robinBaumgarten.Agent(), getLevel(levelFile.getPath()), 30, 0, false);
                GameStatus status = result.getGameStatus();
                if (status == GameStatus.WIN){
                    wins++;
                }
                System.out.println(levelFile.getName() + " Completed");
            }
            float winPercent = (float) wins;
            winPercent = winPercent/ listOfFiles.length;
            System.out.println("Completable: " + wins + " : Completable Percent: " + winPercent);
        }catch (Exception e){
            System.out.println(e);
        }
    }
}
